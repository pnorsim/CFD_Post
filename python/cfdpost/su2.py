"""SU2 native meshes: points, volume elements and boundary markers."""
from __future__ import annotations

import hashlib
import os

import numpy as np

from . import _core
from .grid import Cells, Grid


def marker_list(path: str, zone: int = 1) -> dict:
    """{marker: number of boundary elements} without reading points/elements."""
    return dict(_core.su2_marker_list(os.fspath(path), zone))


def read_su2(path: str, elements: bool = True, points: bool = True, markers=True, zone: int = 1) -> "Su2Mesh":
    """Read an SU2 mesh (multithreaded).

    elements: read volume elements (skip for surface-only work: much less memory)
    markers : True for all, False for none, or a list of marker names
    """
    only = [] if markers in (True, False, None) else list(markers)
    d = _core.read_su2(os.fspath(path), points, elements, markers is not False, only, zone)
    return Su2Mesh(ndim=d["ndim"], npoin=d["npoin"], nelem=d["nelem"],
                   points=d["points"] if points else None,
                   elements=Cells.from_dict(d["elements"]) if elements else None,
                   markers={k: Cells.from_dict(v) for k, v in d["markers"].items()},
                   path=os.fspath(path))


def _element_lines(cells: Cells, index_column: bool) -> list:
    sizes = cells.sizes
    out = [None] * len(cells)
    for k in np.unique(sizes):
        sel = np.flatnonzero(sizes == k)
        rows = np.column_stack([cells.types[sel].astype(np.int64),
                                cells.conn[cells.offsets[sel][:, None] + np.arange(k)]])
        if index_column:
            rows = np.column_stack([rows, sel])
        for i, r in zip(sel.tolist(), rows.tolist()):
            out[i] = "\t".join(map(str, r))
    return out


def write_su2(path, points, elements: Cells, markers: dict, ndim: int = 3, index_column: bool = True):
    """Write a native SU2 ASCII mesh (single zone).

    points  : (n, 3) coordinates (z ignored for ndim=2)
    elements: volume elements (VTK/SU2 type codes)
    markers : {name: Cells} boundary elements, global point ids
    """
    points = np.asarray(points, np.float64)
    with open(os.fspath(path), "w") as f:
        f.write(f"NDIME= {ndim}\nNELEM= {len(elements)}\n")
        f.write("\n".join(_element_lines(elements, index_column)) + ("\n" if len(elements) else ""))
        f.write(f"NPOIN= {len(points)}\n")
        cols = [points[:, d] for d in range(ndim)]
        fmt = "\t".join(["%.17g"] * ndim)
        if index_column:
            cols.append(np.arange(len(points)))
            fmt += "\t%d"
        np.savetxt(f, np.column_stack(cols), fmt=fmt)
        f.write(f"NMARK= {len(markers)}\n")
        for name, c in markers.items():
            f.write(f"MARKER_TAG= {name}\nMARKER_ELEMS= {len(c)}\n")
            if len(c):
                f.write("\n".join(_element_lines(c, False)) + "\n")


class Su2Mesh:
    """SU2 mesh. Marker cells and volume elements use global point ids.

    Two meshes with the same topology (e.g. baseline and SU2_DEF output) differ
    only in ``points``; see :mod:`cfdpost.deform`.
    """

    def __init__(self, ndim, npoin, nelem, points, elements, markers, path=None):
        self.ndim, self.npoin, self.nelem = ndim, npoin, nelem
        self.points = points
        self.elements = elements
        self.markers: dict[str, Cells] = markers
        self.path = path

    @classmethod
    def read(cls, path, **kw) -> "Su2Mesh":
        return read_su2(path, **kw)

    def __repr__(self):
        return (f"Su2Mesh({self.path!r}, ndim={self.ndim}, npoin={self.npoin}, nelem={self.nelem}, "
                f"markers={list(self.markers)})")

    @property
    def marker_names(self):
        return list(self.markers)

    def _names(self, names):
        if names is None:
            return self.marker_names
        if isinstance(names, str):
            names = [names]
        missing = [n for n in names if n not in self.markers]
        if missing:
            raise KeyError(f"markers not in mesh: {missing}; available: {self.marker_names}")
        return list(names)

    def faces(self, names=None) -> tuple[Cells, np.ndarray]:
        """Boundary faces of the given markers (global ids) and the marker index
        (position in `names`) of every face."""
        names = self._names(names)
        cells, tag = Cells.empty(), []
        for i, n in enumerate(names):
            cells = cells.concat(self.markers[n])
            tag.append(np.full(len(self.markers[n]), i, np.int32))
        return cells, (np.concatenate(tag) if tag else np.zeros(0, np.int32))

    def boundary_nodes(self, names=None) -> np.ndarray:
        """Sorted unique global ids of the nodes of the given markers."""
        names = self._names(names)
        return np.unique(np.concatenate([self.markers[n].conn for n in names])) if names else np.zeros(0, np.int64)

    def surface(self, names=None, points=None) -> Grid:
        """Markers as a compact surface Grid; point array 'GlobalIndex', cell array
        'MarkerID' (index into field_data['MarkerNames'])."""
        names = self._names(names)
        pts = self.points if points is None else np.asarray(points)
        if pts is None:
            raise ValueError("mesh was read without points")
        faces, tag = self.faces(names)
        g = Grid(pts, faces, cell_data={"MarkerID": tag}).extract_cells(slice(None))
        g.point_data["GlobalIndex"] = g.point_data.pop("vtkOriginalPointIds")
        g.field_data["MarkerNames"] = names
        return g

    def volume(self) -> Grid:
        if self.elements is None or self.points is None:
            raise ValueError("mesh was read without elements or points")
        return Grid(self.points, self.elements)

    def face_orientation(self, names=None) -> np.ndarray:
        """+1 where a face's right-hand normal points into the fluid (into the
        adjacent volume element), -1 where it points out, 0 if no element found.
        Needs volume elements and points."""
        if self.elements is None or self.points is None:
            raise ValueError("face_orientation needs a mesh read with elements and points")
        faces, _ = self.faces(names)
        e = self.elements
        return _core.orient_faces(self.points, e.types, e.offsets, e.conn, faces.types, faces.offsets, faces.conn)

    # ------------------------------------------------------------ topology
    def topology_hash(self) -> str:
        """Hash of element and marker connectivity (not of coordinates)."""
        h = hashlib.sha1()
        h.update(np.array([self.ndim, self.npoin, self.nelem], np.int64).tobytes())
        if self.elements is not None:
            h.update(self.elements.types.tobytes())
            h.update(self.elements.conn.tobytes())
        for k in sorted(self.markers):
            h.update(k.encode())
            h.update(self.markers[k].conn.tobytes())
        return h.hexdigest()

    def same_topology(self, other: "Su2Mesh", check_elements: bool = True) -> bool:
        if (self.ndim, self.npoin) != (other.ndim, other.npoin) or set(self.markers) != set(other.markers):
            return False
        for k, m in self.markers.items():
            o = other.markers[k]
            if len(m) != len(o) or not np.array_equal(m.conn, o.conn):
                return False
        if check_elements and self.elements is not None and other.elements is not None:
            return (len(self.elements) == len(other.elements) and
                    np.array_equal(self.elements.types, other.elements.types) and
                    np.array_equal(self.elements.conn, other.elements.conn))
        return True

    def write(self, path, index_column: bool = True):
        """Write as a native SU2 mesh (needs points and elements)."""
        if self.points is None or self.elements is None:
            raise ValueError("writing needs a mesh read with points and elements")
        write_su2(path, self.points, self.elements, self.markers, self.ndim, index_column)

    def with_points(self, points) -> "Su2Mesh":
        """Same topology, new coordinates (shares connectivity arrays)."""
        points = np.ascontiguousarray(points, dtype=np.float64)
        if points.shape != (self.npoin, 3):
            raise ValueError(f"expected points of shape ({self.npoin}, 3)")
        return Su2Mesh(self.ndim, self.npoin, self.nelem, points, self.elements, self.markers, self.path)
