"""Surface solutions (SU2 surface_flow.vtu or markers extracted from a volume
solution) with boundary markers attached."""
from __future__ import annotations

import os

import numpy as np

from .grid import Cells, Grid
from .io import VtkFile
from .su2 import Su2Mesh, read_su2


def face_keys(cells: Cells, ids: np.ndarray | None = None) -> np.ndarray:
    """Order-independent key of every face: sorted node ids padded with -1, as
    a void view usable with np.unique / searchsorted."""
    conn = cells.conn if ids is None else ids[cells.conn]
    sizes = cells.sizes
    if len(sizes) and sizes.max() > 4:
        raise ValueError("faces with more than 4 nodes are not supported for marker matching")
    k = np.full((len(cells), 4), -1, np.int64)
    pos = np.arange(len(conn)) - np.repeat(cells.offsets[:-1], sizes)
    k[np.repeat(np.arange(len(cells)), sizes), pos] = conn
    k.sort(axis=1)
    return np.ascontiguousarray(k).view(np.dtype((np.void, 32))).ravel()


class SurfaceSolution:
    """Surface grid + point data, optionally tagged with SU2 markers.

    Attributes
    ----------
    grid        : Grid (points, faces, point/cell data)
    global_ids  : global SU2 point id of every surface point (None if unknown)
    marker_id   : per face, index into marker_names (-1 if untagged)
    marker_names: list of marker names
    """

    def __init__(self, grid: Grid, global_ids=None, marker_id=None, marker_names=None, path=None):
        self.grid = grid
        self.global_ids = None if global_ids is None else np.asarray(global_ids, np.int64)
        self.marker_names = list(marker_names or [])
        self.marker_id = (np.full(grid.n_cells, -1, np.int32) if marker_id is None
                          else np.asarray(marker_id, np.int32))
        self.path = path
        self._sign = None

    def __repr__(self):
        return (f"SurfaceSolution({self.grid.n_points} points, {self.grid.n_cells} faces, "
                f"markers={self.marker_names}, arrays={list(self.grid.point_data)})")

    # ------------------------------------------------------------ construction
    @classmethod
    def read(cls, path, mesh: Su2Mesh | str | None = None, arrays=None, plotted=None,
             match_tol: float = 1e-4) -> "SurfaceSolution":
        """Read an SU2 surface solution (surface_flow.vtu).

        mesh    : Su2Mesh (or path) providing marker tags. SU2 writes the surface
                  points as the nodes of the MARKER_PLOTTING markers in increasing
                  global index; that order is used directly and verified against
                  the marker faces (topologically, so a deformed solution can be
                  tagged with the undeformed mesh). If it does not hold, points
                  are matched by coordinates (within match_tol).
        plotted : marker names of MARKER_PLOTTING (default: inferred)
        arrays  : point arrays to load (default all)
        """
        f = VtkFile(path)
        grid = f.grid(point_arrays=arrays)
        sol = cls(grid, path=os.fspath(path))
        if mesh is not None:
            if not isinstance(mesh, Su2Mesh):
                mesh = read_su2(mesh, elements=False)
            sol.attach_markers(mesh, plotted=plotted, match_tol=match_tol)
        return sol

    @classmethod
    def from_volume(cls, vtu, mesh: Su2Mesh | str, markers, arrays=None,
                    points: str = "solution") -> "SurfaceSolution":
        """Markers of an SU2 mesh with point data taken from a volume solution
        (flow.vtu, points in global index order). Only the marker nodes are read
        from the volume file.

        points: 'solution' (coordinates stored in the VTU, i.e. the geometry the
                solution was computed on) or 'mesh' (coordinates of `mesh`).
        """
        if not isinstance(mesh, Su2Mesh):
            mesh = read_su2(mesh, elements=False, markers=markers if not isinstance(markers, str) else [markers])
        f = vtu if isinstance(vtu, VtkFile) else VtkFile(vtu)
        if f.n_points != mesh.npoin:
            raise ValueError(f"{f.path} has {f.n_points} points, mesh has {mesh.npoin}")
        names = [markers] if isinstance(markers, str) else list(markers)
        faces, tag = mesh.faces(names)
        gids = np.unique(faces.conn)
        loc = np.full(mesh.npoin, -1, np.int64)
        loc[gids] = np.arange(len(gids))
        xyz = f.points(ids=gids) if points == "solution" else mesh.points[gids]
        pd = {n: f.point(n, ids=gids) for n in (f.point_arrays if arrays is None else arrays)}
        if points == "solution" and mesh.points is not None:
            pd["MeshDisplacement"] = xyz - mesh.points[gids]
        grid = Grid(xyz, faces.remap(loc), pd, {"MarkerID": tag})
        return cls(grid, gids, tag, names, path=f.path)

    def attach_markers(self, mesh: Su2Mesh, plotted=None, match_tol: float = 1e-4):
        n = self.grid.n_points
        gids = None
        cands = []
        if plotted is not None:
            cands.append(list(plotted))
        else:
            cands.append(mesh.marker_names)
            lower = [m for m in mesh.marker_names if not any(s in m.lower() for s in ("sym", "far"))]
            cands.append(lower)
        for names in cands:
            nodes = mesh.boundary_nodes(names)
            if len(nodes) == n and self._tag_faces(mesh, nodes):
                gids = nodes
                break
        if gids is None:
            if mesh.points is None:
                raise ValueError("surface points are not the sorted marker nodes and the mesh has no "
                                 "points for coordinate matching")
            from scipy.spatial import cKDTree
            nodes = mesh.boundary_nodes()
            d, j = cKDTree(mesh.points[nodes]).query(self.grid.points)
            if d.max() > match_tol:
                raise ValueError(f"surface points do not match the mesh boundary (max distance {d.max():.3e}); "
                                 "wrong mesh?")
            gids = nodes[j]
            if not self._tag_faces(mesh, gids, strict=False):
                raise ValueError("faces of the surface solution do not match marker faces")
        self.global_ids = gids
        self.grid.point_data["GlobalIndex"] = gids
        self.grid.cell_data["MarkerID"] = self.marker_id
        if mesh.points is not None:
            d = np.linalg.norm(mesh.points[gids] - self.grid.points, axis=1)
            self.mesh_distance = float(d.max())
        return self

    def _tag_faces(self, mesh: Su2Mesh, gids: np.ndarray, strict: bool = True) -> bool:
        keys = face_keys(self.grid.cells, gids)
        names = mesh.marker_names
        mfaces, mtag = mesh.faces(names)
        mkeys = face_keys(mfaces)
        order = np.argsort(mkeys, kind="stable")
        sk = mkeys[order]
        pos = np.searchsorted(sk, keys)
        pos = np.minimum(pos, len(sk) - 1)
        found = sk[pos] == keys
        if strict and not found.all():
            return False
        if found.sum() < 0.99 * len(keys):
            return False
        mid = np.full(len(keys), -1, np.int32)
        mid[found] = mtag[order[pos[found]]]
        self.marker_id, self.marker_names = mid, names
        return True

    # ------------------------------------------------------------ access
    @property
    def points(self) -> np.ndarray:
        return self.grid.points

    @property
    def cells(self) -> Cells:
        return self.grid.cells

    def __getitem__(self, name):
        return self.grid[name]

    def select(self, markers) -> "SurfaceSolution":
        """Sub-surface with the faces of the given markers (points compacted)."""
        names = [markers] if isinstance(markers, str) else list(markers)
        missing = [m for m in names if m not in self.marker_names]
        if missing:
            raise KeyError(f"markers {missing} not present; available: {self.marker_names}")
        ids = [self.marker_names.index(m) for m in names]
        mask = np.isin(self.marker_id, ids)
        g = self.grid.extract_cells(mask)
        orig = g.point_data.pop("vtkOriginalPointIds")
        gids = None if self.global_ids is None else self.global_ids[orig]
        out = SurfaceSolution(g, gids, self.marker_id[mask], self.marker_names, self.path)
        if self._sign is not None:
            out._sign = self._sign[mask]
        return out

    # ------------------------------------------------------------ orientation
    def orientation(self, method: str = "auto", mesh: Su2Mesh | None = None) -> np.ndarray:
        """Per-face +1/-1 turning right-hand normals into normals pointing into the fluid.

        method: 'mesh'   exact, from the adjacent volume element (needs `mesh` with elements)
                'closed' the selection is treated as a (nearly) closed body:
                         one global sign making sum(r . n dA) > 0 (needs consistent faces)
                'auto'   cached result, else 'mesh' if given, else 'closed'
        """
        if method == "auto":
            if self._sign is not None and mesh is None:
                return self._sign
            method = "mesh" if mesh is not None else "closed"
        if method == "mesh":
            if self.global_ids is None:
                raise ValueError("orientation from the mesh needs marker-tagged surfaces")
            g = self.grid
            faces = g.cells.remap(self.global_ids)
            e = mesh.elements
            if e is None:
                raise ValueError("mesh has no volume elements; read it with elements=True")
            from . import _core
            s = _core.orient_faces(mesh.points, e.types, e.offsets, e.conn, faces.types, faces.offsets, faces.conn)
            s = s.astype(np.float64)
            if np.any(s == 0):
                raise ValueError(f"{int(np.sum(s == 0))} faces have no adjacent volume element")
        elif method == "closed":
            S = self.grid.area_vectors()
            c = self.grid.cell_centers()
            s = np.ones(self.grid.n_cells) * (1.0 if np.sum(np.einsum("ij,ij->i", c - c.mean(0), S)) > 0 else -1.0)
        else:
            raise ValueError(method)
        self._sign = s
        return s

    def set_orientation(self, sign):
        self._sign = np.broadcast_to(np.asarray(sign, np.float64), (self.grid.n_cells,)).copy()

    def write(self, path):
        self.grid.write(path)
