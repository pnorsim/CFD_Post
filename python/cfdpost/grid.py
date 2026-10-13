"""In-memory unstructured grid / poly data with point and cell arrays.

Cells use VTK type codes in compressed-row form (``Cells``): cell ``i`` has
nodes ``conn[offsets[i]:offsets[i+1]]``. The same container holds volumes,
surfaces (marker faces), iso-surfaces (polygons) and cuts (line segments).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

VTK_VERTEX, VTK_LINE, VTK_TRIANGLE, VTK_POLYGON, VTK_QUAD = 1, 3, 5, 7, 9
VTK_TETRA, VTK_HEXAHEDRON, VTK_WEDGE, VTK_PYRAMID = 10, 12, 13, 14
CELL_DIM = {1: 0, 3: 1, 5: 2, 7: 2, 9: 2, 10: 3, 12: 3, 13: 3, 14: 3}
CELL_NODES = {1: 1, 3: 2, 5: 3, 9: 4, 10: 4, 12: 8, 13: 6, 14: 5}


@dataclass
class Cells:
    types: np.ndarray            # (n,) uint8 VTK codes
    offsets: np.ndarray          # (n+1,) int64, starts with 0
    conn: np.ndarray             # int64 point ids

    def __post_init__(self):
        self.types = np.ascontiguousarray(self.types, dtype=np.uint8)
        self.offsets = np.ascontiguousarray(self.offsets, dtype=np.int64)
        self.conn = np.ascontiguousarray(self.conn, dtype=np.int64)

    @classmethod
    def from_dict(cls, d) -> "Cells":
        return cls(d["types"], d["offsets"], d["conn"])

    @classmethod
    def empty(cls) -> "Cells":
        return cls(np.zeros(0, np.uint8), np.zeros(1, np.int64), np.zeros(0, np.int64))

    @classmethod
    def uniform(cls, vtk_type: int, conn) -> "Cells":
        """Cells of one fixed-size type from an (n, k) node array."""
        conn = np.asarray(conn, dtype=np.int64)
        n, k = conn.shape
        return cls(np.full(n, vtk_type, np.uint8), np.arange(n + 1, dtype=np.int64) * k, conn.ravel())

    def __len__(self):
        return len(self.types)

    @property
    def sizes(self) -> np.ndarray:
        return np.diff(self.offsets)

    def cell(self, i):
        return self.conn[self.offsets[i]:self.offsets[i + 1]]

    def subset(self, mask_or_index) -> "Cells":
        idx = np.arange(len(self))[mask_or_index]
        sizes = self.sizes[idx]
        offsets = np.zeros(len(idx) + 1, np.int64)
        np.cumsum(sizes, out=offsets[1:])
        starts = self.offsets[idx]
        # gather variable-length rows
        rep = np.repeat(starts - offsets[:-1], sizes)
        conn = self.conn[np.arange(offsets[-1]) + rep]
        return Cells(self.types[idx], offsets, conn)

    def concat(self, other: "Cells") -> "Cells":
        return Cells(np.concatenate([self.types, other.types]),
                     np.concatenate([self.offsets, other.offsets[1:] + self.offsets[-1]]),
                     np.concatenate([self.conn, other.conn]))

    def remap(self, mapping: np.ndarray) -> "Cells":
        return Cells(self.types, self.offsets, mapping[self.conn])

    def cell_ids_of_nodes(self) -> np.ndarray:
        """Cell index of every entry in conn."""
        return np.repeat(np.arange(len(self)), self.sizes)

    def triangles(self) -> tuple[np.ndarray, np.ndarray]:
        """Fan-triangulate 2D cells: (tris (m,3), parent cell (m,))."""
        sizes = self.sizes
        tris, parent = [], []
        for k in np.unique(sizes[np.isin(self.types, (VTK_TRIANGLE, VTK_QUAD, VTK_POLYGON))]):
            sel = np.flatnonzero((sizes == k) & np.isin(self.types, (VTK_TRIANGLE, VTK_QUAD, VTK_POLYGON)))
            rows = self.conn[self.offsets[sel][:, None] + np.arange(k)]
            for j in range(1, k - 1):
                tris.append(rows[:, [0, j, j + 1]])
                parent.append(sel)
        if not tris:
            return np.zeros((0, 3), np.int64), np.zeros(0, np.int64)
        return np.vstack(tris), np.concatenate(parent)

    def edges(self) -> np.ndarray:
        """Unique undirected edges (m, 2) of 1D/2D cells (cycle edges of polygons)."""
        sizes = self.sizes
        e = []
        for k in np.unique(sizes):
            sel = np.flatnonzero(sizes == k)
            rows = self.conn[self.offsets[sel][:, None] + np.arange(k)]
            if k == 2:
                e.append(rows)
            else:
                e.append(np.stack([rows, np.roll(rows, -1, axis=1)], -1).reshape(-1, 2))
        if not e:
            return np.zeros((0, 2), np.int64)
        e = np.sort(np.vstack(e), axis=1)
        return np.unique(e, axis=0)


@dataclass
class Grid:
    points: np.ndarray
    cells: Cells
    point_data: dict = field(default_factory=dict)
    cell_data: dict = field(default_factory=dict)
    field_data: dict = field(default_factory=dict)

    def __post_init__(self):
        self.points = np.ascontiguousarray(self.points, dtype=np.float64).reshape(-1, 3)

    def __repr__(self):
        return (f"Grid({self.n_points} points, {self.n_cells} cells, "
                f"point_data={list(self.point_data)}, cell_data={list(self.cell_data)})")

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    def __getitem__(self, name):
        if name in self.point_data:
            return self.point_data[name]
        if name in self.cell_data:
            return self.cell_data[name]
        raise KeyError(name)

    # ------------------------------------------------------------ geometry
    def cell_centers(self) -> np.ndarray:
        if self.n_cells == 0:
            return np.zeros((0, 3))
        c = np.add.reduceat(self.points[self.cells.conn], self.cells.offsets[:-1], axis=0)
        return c / self.cells.sizes[:, None]

    def area_vectors(self) -> np.ndarray:
        """Area vector of each 2D cell (Newell's formula; right-hand rule)."""
        if self.n_cells == 0:
            return np.zeros((0, 3))
        nxt = np.arange(len(self.cells.conn)) + 1
        nxt[self.cells.offsets[1:] - 1] = self.cells.offsets[:-1]
        a = self.points[self.cells.conn]
        b = self.points[self.cells.conn[nxt]]
        return 0.5 * np.add.reduceat(np.cross(a, b), self.cells.offsets[:-1], axis=0)

    def areas(self) -> np.ndarray:
        return np.linalg.norm(self.area_vectors(), axis=1)

    def bounds(self) -> np.ndarray:
        return np.array([self.points.min(0), self.points.max(0)])

    # ------------------------------------------------------------ selection
    def extract_cells(self, mask_or_index, compact: bool = True) -> "Grid":
        """Sub-grid of the selected cells (points compacted, data carried over).
        The original point ids are kept as point array 'vtkOriginalPointIds'."""
        idx = np.arange(self.n_cells)[mask_or_index]
        cells = self.cells.subset(idx)
        cd = {k: v[idx] for k, v in self.cell_data.items()}
        if not compact:
            return Grid(self.points, cells, dict(self.point_data), cd, dict(self.field_data))
        used = np.unique(cells.conn)
        remap = np.full(self.n_points, -1, np.int64)
        remap[used] = np.arange(len(used))
        pd = {k: v[used] for k, v in self.point_data.items()}
        pd["vtkOriginalPointIds"] = used
        return Grid(self.points[used], cells.remap(remap), pd, cd, dict(self.field_data))

    def threshold(self, name: str, lower=-np.inf, upper=np.inf, all_points: bool = True) -> "Grid":
        """Keep cells whose point values of `name` lie in [lower, upper]: all points
        (all_points=True, like VTK AllScalars) or any point. Cell arrays are
        thresholded directly."""
        if name in self.cell_data:
            v = self.cell_data[name]
            return self.extract_cells((v >= lower) & (v <= upper))
        v = self.point_data[name]
        if v.ndim > 1:
            v = np.linalg.norm(v, axis=1)
        if self.n_cells == 0:
            return self.extract_cells(slice(None))
        ok = ((v >= lower) & (v <= upper))[self.cells.conn]
        red = np.logical_and if all_points else np.logical_or
        return self.extract_cells(red.reduceat(ok, self.cells.offsets[:-1]))

    def connected_components(self) -> np.ndarray:
        """Region id per cell (cells sharing a point are connected)."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        ids = self.cells.cell_ids_of_nodes()
        n = self.n_cells
        # bipartite graph cells <-> points
        g = coo_matrix((np.ones(len(ids)), (ids, n + self.cells.conn)), shape=(n + self.n_points,) * 2)
        _, lab = connected_components(g, directed=False)
        cell_lab = lab[:n]
        _, cell_lab = np.unique(cell_lab, return_inverse=True)
        return cell_lab

    def largest_component(self) -> "Grid":
        lab = self.connected_components()
        return self.extract_cells(lab == np.bincount(lab).argmax())

    def clean(self, tol: float = 0.0) -> "Grid":
        """Merge coincident points (exactly equal, or within tol) and drop unused ones."""
        if tol > 0:
            from scipy.spatial import cKDTree
            pairs = cKDTree(self.points).query_pairs(tol, output_type="ndarray")
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
            g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(self.n_points,) * 2)
            _, rep = connected_components(g, directed=False)
            _, first = np.unique(rep, return_index=True)
            inv = rep
            uniq_pts = self.points[first]
            keep_idx = first
        else:
            uniq_pts, keep_idx, inv = np.unique(self.points, axis=0, return_index=True, return_inverse=True)
        cells = self.cells.remap(inv.ravel())
        pd = {k: v[keep_idx] for k, v in self.point_data.items()}
        return Grid(uniq_pts, cells, pd, dict(self.cell_data), dict(self.field_data)).extract_cells(slice(None))

    def write(self, path: str):
        from .io import write_vtk
        write_vtk(path, self)

    def to_pyvista(self):
        import pyvista as pv
        sizes = self.cells.sizes
        legacy = np.insert(self.cells.conn, self.cells.offsets[:-1], sizes)
        g = pv.UnstructuredGrid(legacy, self.cells.types, self.points)
        for k, v in self.point_data.items():
            g.point_data[k] = v
        for k, v in self.cell_data.items():
            g.cell_data[k] = v
        return g
