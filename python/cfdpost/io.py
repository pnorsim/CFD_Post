"""File I/O: VTK XML (.vtu/.vtp) through the C++ memory-mapped reader, and a
fast raw-appended VTK XML writer."""
from __future__ import annotations

import os

import numpy as np

from . import _core
from .grid import CELL_DIM, VTK_LINE, VTK_VERTEX, Cells, Grid

_VTK_TYPES = {np.dtype(np.int8): "Int8", np.dtype(np.uint8): "UInt8", np.dtype(np.int16): "Int16",
              np.dtype(np.uint16): "UInt16", np.dtype(np.int32): "Int32", np.dtype(np.uint32): "UInt32",
              np.dtype(np.int64): "Int64", np.dtype(np.uint64): "UInt64", np.dtype(np.float32): "Float32",
              np.dtype(np.float64): "Float64"}


class VtkFile:
    """Lazy handle on a VTK XML file: header parsed, arrays read on demand.

    >>> f = VtkFile("flow.vtu")
    >>> f.point_arrays           # names
    >>> p = f.point("Pressure")  # whole array (stored dtype)
    >>> p = f.point("Pressure", ids=nodes)  # gathered tuples, float64
    """

    def __init__(self, path: str):
        self.path = os.fspath(path)
        self._f = _core.VtkXmlFile(self.path)

    def __repr__(self):
        return f"VtkFile({self.path!r}, {self.dataset_type}, {self.n_points} points, {self.n_cells} cells)"

    @property
    def dataset_type(self) -> str:
        return self._f.dataset_type

    @property
    def n_points(self) -> int:
        return self._f.num_points

    @property
    def n_cells(self) -> int:
        return self._f.num_cells

    def arrays(self, section=None):
        a = self._f.arrays()
        return [x for x in a if section is None or x["section"] == section]

    @property
    def point_arrays(self):
        return [a["name"] for a in self.arrays("PointData")]

    @property
    def cell_arrays(self):
        return [a["name"] for a in self.arrays("CellData")]

    def read(self, section, name, first=0, count=-1, ids=None):
        if ids is not None:
            return self._f.read_tuples(section, name, np.asarray(ids, dtype=np.int64))
        return self._f.read(section, name, first, count)

    def point(self, name, ids=None, dtype=None):
        a = self.read("PointData", name, ids=ids)
        return a if dtype is None else a.astype(dtype, copy=False)

    def cell(self, name, ids=None, dtype=None):
        a = self.read("CellData", name, ids=ids)
        return a if dtype is None else a.astype(dtype, copy=False)

    def points(self, ids=None, dtype=np.float64):
        a = self.read("Points", "Points", ids=ids)
        return a.astype(dtype, copy=False)

    def cells(self) -> Cells:
        if self.dataset_type == "UnstructuredGrid":
            ends = self.read("Cells", "offsets").astype(np.int64)
            return Cells(self.read("Cells", "types"), np.concatenate([[0], ends]),
                         self.read("Cells", "connectivity").astype(np.int64))
        # PolyData: verts, lines, polys (VTK cell order)
        out = Cells.empty()
        for sec, code in (("Verts", VTK_VERTEX), ("Lines", VTK_LINE), ("Polys", None)):
            if not self.arrays(sec):
                continue
            ends = self.read(sec, "offsets").astype(np.int64)
            conn = self.read(sec, "connectivity").astype(np.int64)
            offs = np.concatenate([[0], ends])
            sizes = np.diff(offs)
            if code is None:
                types = np.where(sizes == 3, 5, np.where(sizes == 4, 9, 7)).astype(np.uint8)
            else:
                types = np.full(len(sizes), code, np.uint8)
            out = out.concat(Cells(types, offs, conn))
        return out

    def grid(self, point_arrays=None, cell_arrays=None, dtype=np.float64) -> Grid:
        """Whole dataset as a Grid (selected arrays; all by default)."""
        pa = self.point_arrays if point_arrays is None else list(point_arrays)
        ca = self.cell_arrays if cell_arrays is None else list(cell_arrays)
        pd = {n: self.point(n).astype(dtype, copy=False) for n in pa}
        cd = {n: self.cell(n).astype(dtype, copy=False) for n in ca}
        return Grid(self.points(), self.cells(), pd, cd)


def read_vtk(path: str, point_arrays=None, cell_arrays=None) -> Grid:
    """Read a .vtu / .vtp file into a Grid. Legacy .vtk files go through VTK."""
    path = os.fspath(path)
    if path.endswith(".vtk"):
        return _read_with_vtk(path)
    return VtkFile(path).grid(point_arrays, cell_arrays)


def _read_with_vtk(path: str) -> Grid:
    import pyvista as pv
    m = pv.read(path)
    if not isinstance(m, pv.UnstructuredGrid):
        m = m.cast_to_unstructured_grid()
    conn = m.cell_connectivity.astype(np.int64)
    offs = m.offset.astype(np.int64)
    return Grid(np.asarray(m.points, float), Cells(m.celltypes, offs, conn),
                {k: np.asarray(v) for k, v in m.point_data.items()},
                {k: np.asarray(v) for k, v in m.cell_data.items()})


# --------------------------------------------------------------------------- writer
class _Appended:
    def __init__(self):
        self.blocks, self.offset = [], 0

    def add(self, arr) -> str:
        arr = np.ascontiguousarray(arr)
        if arr.dtype == np.bool_:
            arr = arr.astype(np.uint8)
        if arr.dtype not in _VTK_TYPES:
            arr = arr.astype(np.float64)
        nb = arr.nbytes
        self.blocks.append((np.uint64(nb).tobytes(), arr))
        off = self.offset
        self.offset += 8 + nb
        return f'type="{_VTK_TYPES[arr.dtype]}" format="appended" offset="{off}"'


def _data_array(app, name, arr):
    arr = np.asarray(arr)
    nc = 1 if arr.ndim == 1 else int(np.prod(arr.shape[1:]))
    return f'<DataArray Name="{name}" NumberOfComponents="{nc}" {app.add(arr.reshape(len(arr), -1) if arr.ndim > 1 else arr)}/>'


def write_vtk(path: str, grid: Grid, binary_float32: bool = False):
    """Write a Grid as .vtu (UnstructuredGrid) or .vtp (PolyData: 0D/1D/2D cells).
    Raw appended data with UInt64 headers, readable by ParaView/VTK and by
    cfdpost's own reader without copies."""
    path = os.fspath(path)
    poly = path.endswith(".vtp")
    app = _Appended()
    cast = (lambda a: a.astype(np.float32) if binary_float32 and a.dtype == np.float64 else a)
    lines = ['<?xml version="1.0"?>',
             f'<VTKFile type="{"PolyData" if poly else "UnstructuredGrid"}" version="1.0" '
             'byte_order="LittleEndian" header_type="UInt64">',
             f'<{"PolyData" if poly else "UnstructuredGrid"}>']
    cells = grid.cells
    cell_order = np.arange(len(cells))
    if poly:
        dims = np.array([CELL_DIM.get(int(t), 2) for t in cells.types]) if len(cells) else np.zeros(0, int)
        if np.any(dims == 3):
            raise ValueError("PolyData cannot hold 3D cells; write .vtu")
        cell_order = np.argsort(dims, kind="stable")       # verts, lines, polys
        groups = {"Verts": cell_order[dims[cell_order] == 0], "Lines": cell_order[dims[cell_order] == 1],
                  "Polys": cell_order[dims[cell_order] == 2]}
        counts = " ".join(f'NumberOf{k}="{len(v)}"' for k, v in groups.items())
        lines.append(f'<Piece NumberOfPoints="{grid.n_points}" {counts} NumberOfStrips="0">')
    else:
        lines.append(f'<Piece NumberOfPoints="{grid.n_points}" NumberOfCells="{len(cells)}">')
    lines.append("<PointData>")
    for k, v in grid.point_data.items():
        lines.append(_data_array(app, k, cast(np.asarray(v))))
    lines.append("</PointData><CellData>")
    for k, v in grid.cell_data.items():
        lines.append(_data_array(app, k, cast(np.asarray(v))[cell_order]))
    lines.append("</CellData>")
    lines.append(f'<Points><DataArray NumberOfComponents="3" {app.add(cast(grid.points))}/></Points>')
    if poly:
        for sec, idx in groups.items():
            sub = cells.subset(idx)
            lines.append(f"<{sec}>")
            lines.append(f'<DataArray Name="connectivity" {app.add(sub.conn)}/>')
            lines.append(f'<DataArray Name="offsets" {app.add(sub.offsets[1:])}/>')
            lines.append(f"</{sec}>")
    else:
        lines.append("<Cells>")
        lines.append(f'<DataArray Name="connectivity" {app.add(cells.conn)}/>')
        lines.append(f'<DataArray Name="offsets" {app.add(cells.offsets[1:])}/>')
        lines.append(f'<DataArray Name="types" {app.add(cells.types)}/>')
        lines.append("</Cells>")
    lines.append("</Piece>")
    if grid.field_data:
        pass  # field data is not written (VTK XML stores it outside Piece; rarely needed)
    lines.append(f'</{"PolyData" if poly else "UnstructuredGrid"}>')
    lines.append('<AppendedData encoding="raw">')
    with open(path, "wb") as f:
        f.write(("\n".join(lines) + "\n_").encode())
        for head, arr in app.blocks:
            f.write(head)
            arr.tofile(f)
        f.write(b"\n</AppendedData>\n</VTKFile>\n")
