"""Meshes and solutions that share topology but not node positions.

Typical SU2 optimisation post-processing: a baseline mesh and the SU2_DEF
output have identical elements and markers, only the coordinates move. Nodes
are matched by global index (no geometric search), so baseline and optimised
solutions can be compared point by point on either geometry.
"""
from __future__ import annotations

import numpy as np

from .grid import Grid
from .io import VtkFile
from .su2 import Su2Mesh, read_su2
from .surface import SurfaceSolution


def coordinates(source, npoin: int | None = None) -> np.ndarray:
    """Global coordinates from an Su2Mesh, an .su2 file (points only are read),
    a volume .vtu in global point order, or an array."""
    if isinstance(source, Su2Mesh):
        xyz = source.points
    elif isinstance(source, np.ndarray):
        xyz = source
    elif str(source).endswith(".su2"):
        xyz = read_su2(source, elements=False, markers=False).points
    else:
        xyz = VtkFile(source).points()
    xyz = np.asarray(xyz, np.float64)
    if npoin is not None and len(xyz) != npoin:
        raise ValueError(f"{source}: {len(xyz)} points, expected {npoin}")
    return xyz


class MeshPair:
    """Baseline topology with two sets of coordinates.

    base     : Su2Mesh (markers needed; read elements=False for surface work)
    deformed : Su2Mesh, .su2 path, volume .vtu (points in global order) or array
    check    : verify same point count and, for Su2Mesh inputs, same markers / elements
    """

    def __init__(self, base, deformed, check: bool = True):
        self.base = base if isinstance(base, Su2Mesh) else read_su2(base, elements=False)
        if isinstance(deformed, Su2Mesh) and check and not self.base.same_topology(deformed):
            raise ValueError("meshes do not have the same topology")
        self.xyz_def = coordinates(deformed, self.base.npoin)
        self.xyz_base = self.base.points

    @property
    def displacement(self) -> np.ndarray:
        return self.xyz_def - self.xyz_base

    def deformed_mesh(self) -> Su2Mesh:
        return self.base.with_points(self.xyz_def)

    def surface(self, markers=None, on: str = "deformed") -> Grid:
        """Marker surface with point arrays Displacement, |Displacement| and
        NormalDisplacement (along the area-weighted point normal of the baseline
        geometry, sign as the faces' right-hand rule)."""
        g = self.base.surface(markers, self.xyz_def if on == "deformed" else self.xyz_base)
        gid = g.point_data["GlobalIndex"]
        d = self.displacement[gid]
        base = Grid(self.xyz_base[gid], g.cells)
        n = point_normals(base)
        g.point_data["Displacement"] = d
        g.point_data["DisplacementMagnitude"] = np.linalg.norm(d, axis=1)
        g.point_data["NormalDisplacement"] = np.einsum("ij,ij->i", d, n)
        return g

    def summary(self, markers=None) -> dict:
        ids = self.base.boundary_nodes(markers)
        d = np.linalg.norm(self.displacement[ids], axis=1)
        dv = np.linalg.norm(self.displacement, axis=1)
        return {"surface_max": float(d.max()), "surface_mean": float(d.mean()),
                "volume_max": float(dv.max()), "moved_nodes": int(np.count_nonzero(dv > 0))}


def point_normals(g: Grid) -> np.ndarray:
    """Unit area-weighted point normals of a surface (right-hand rule of faces)."""
    S = g.area_vectors()
    n = np.zeros((g.n_points, 3))
    np.add.at(n, g.cells.conn, np.repeat(S, g.cells.sizes, axis=0))
    return n / np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]


def compare(sol_a: SurfaceSolution, sol_b: SurfaceSolution, fields=None, on: str = "b",
            labels=("base", "opt")) -> SurfaceSolution:
    """Point-by-point comparison of two marker-tagged surface solutions on the
    same topology. Points are matched by global index. The result carries
    '<f>_<label_a>', '<f>_<label_b>' and 'delta_<f>' (= b - a), plus the
    displacement from a to b, on the geometry of `on` ('a' or 'b')."""
    if sol_a.global_ids is None or sol_b.global_ids is None:
        raise ValueError("both solutions need global ids (read them with an Su2Mesh)")
    common, ia, ib = np.intersect1d(sol_a.global_ids, sol_b.global_ids, assume_unique=True, return_indices=True)
    if len(common) != len(sol_b.global_ids):
        raise ValueError("solutions do not contain the same surface points")
    ref = sol_b if on == "b" else sol_a
    # reorder a onto ref's point order
    pos_ref = ib if ref is sol_b else ia
    idx_a = np.empty(ref.grid.n_points, np.int64)
    idx_b = np.empty(ref.grid.n_points, np.int64)
    idx_a[pos_ref] = ia
    idx_b[pos_ref] = ib
    names = fields or [k for k in sol_a.grid.point_data if k in sol_b.grid.point_data
                       and k not in ("GlobalIndex",)]
    pd = {"GlobalIndex": ref.global_ids}
    la, lb = labels
    for k in names:
        va, vb = np.asarray(sol_a.grid.point_data[k])[idx_a], np.asarray(sol_b.grid.point_data[k])[idx_b]
        pd[f"{k}_{la}"], pd[f"{k}_{lb}"], pd[f"delta_{k}"] = va, vb, vb - va
    pd["Displacement"] = sol_b.grid.points[idx_b] - sol_a.grid.points[idx_a]
    g = Grid(ref.grid.points, ref.grid.cells, pd, dict(ref.grid.cell_data))
    return SurfaceSolution(g, ref.global_ids, ref.marker_id, ref.marker_names)
