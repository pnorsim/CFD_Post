"""Distributed volume solutions.

A ``Volume`` splits the cells of a .vtu over the ranks of a communicator.
Point data are handled as *owned* arrays: rank r holds global points
``own_range = [b, e)`` (every point exactly once), so element-wise numpy code is
identical in serial and parallel, and global statistics go through
``vol.comm`` (sum, quantile, ...). Cell kernels (gradients, contours, slices)
run in C++ with OpenMP inside each rank.

    vol = Volume("flow.vtu")                 # serial or under mpirun
    p = vol.read("Pressure")                 # owned array
    gp = vol.gradient(p)                     # (n_owned, 3)
    iso = vol.contour(mn, 1.0, {"gradP": np.linalg.norm(gp, axis=1)})
    if vol.comm.is_root: iso.write("shock.vtp")
"""
from __future__ import annotations

import os

import numpy as np

from . import _core
from .grid import Cells, Grid
from .mpi import Comm, get_comm


class Volume:
    def __init__(self, source, comm=None):
        """source: path of a .vtu (UnstructuredGrid) or a Grid (every rank passes
        the full grid; each keeps its share)."""
        self.comm: Comm = get_comm(comm)
        if isinstance(source, Grid):
            c = source.cells
            self._p = _core.VolumePartition(source.points, c.types, c.offsets, c.conn, self.comm.core)
            self._grid_pd = source.point_data
            self.path = None
        else:
            self.path = os.fspath(source)
            self._p = _core.VolumePartition(self.path, self.comm.core)
            self._grid_pd = None
        b, e = self._p.own_range
        self.own_slice = slice(b, e)
        self._xyz = None
        self._cache = {}

    def __repr__(self):
        return (f"Volume({self.path!r}, {self.n_points} points, {self.n_cells} cells, rank {self.comm.rank}/"
                f"{self.comm.size}: cells {self._p.cell_range}, owned points {self._p.own_range})")

    # ------------------------------------------------------------ sizes
    @property
    def n_points(self) -> int:
        return self._p.n_points

    @property
    def n_cells(self) -> int:
        return self._p.n_cells

    @property
    def n_owned(self) -> int:
        return self._p.n_owned

    @property
    def dim(self) -> int:
        return self._p.dim

    @property
    def own_range(self):
        return self._p.own_range

    @property
    def partition(self):
        return self._p

    # ------------------------------------------------------------ data
    @property
    def point_arrays(self):
        return list(self._grid_pd) if self._grid_pd is not None else self._p.point_array_names()

    def read(self, name: str, cache: bool = False) -> np.ndarray:
        """Owned slice of a point array, float64."""
        if name in self._cache:
            return self._cache[name]
        if self._grid_pd is not None:
            a = np.asarray(self._grid_pd[name], np.float64)[self.own_slice]
        else:
            a = self._p.read_point_data(name, True)
        if cache:
            self._cache[name] = a
        return a

    def read_touched(self, name: str) -> np.ndarray:
        if self._grid_pd is not None:
            return np.asarray(self._grid_pd[name], np.float64)[self._p.touched]
        return self._p.read_point_data(name, False)

    @property
    def xyz(self) -> np.ndarray:
        """Owned coordinates (n_owned, 3)."""
        if self._xyz is None:
            self._xyz = self._p.xyz_owned()
        return self._xyz

    def set_coordinates(self, xyz_owned):
        """Use other coordinates with the same topology (e.g. a deformed mesh).
        Accepts owned (n_owned, 3) or global (n_points, 3) arrays."""
        xyz = np.asarray(xyz_owned, np.float64)
        if xyz.shape[0] == self.n_points and self.n_points != self.n_owned:
            xyz = xyz[self.own_slice]
        self._p.set_coordinates(np.ascontiguousarray(xyz))
        self._xyz = None

    def owned(self, global_array) -> np.ndarray:
        """Owned slice of a global per-point array."""
        return np.asarray(global_array)[self.own_slice]

    def gather(self, owned, root: int = 0):
        """Assemble a global array on root (None elsewhere). Memory: full size on root."""
        parts = self.comm.gather(np.asarray(owned), root)
        return None if parts is None else np.concatenate(parts)

    # ------------------------------------------------------------ kernels
    def to_touched(self, owned) -> np.ndarray:
        return self._p.to_touched(np.ascontiguousarray(owned, np.float64))

    def gradient(self, owned) -> np.ndarray:
        """Nodal gradient (Green-Gauss per cell, volume-weighted to nodes) of an
        owned array (n_owned, ...) -> (n_owned, ..., 3)."""
        return self._p.gradient(np.ascontiguousarray(owned, np.float64))

    def nodal_volume(self) -> np.ndarray:
        if "__nodal_volume" not in self._cache:
            self._cache["__nodal_volume"] = self._p.nodal_volume()
        return self._cache["__nodal_volume"]

    def length_scale(self) -> np.ndarray:
        """Local mesh size h = (dual volume)^(1/dim), owned."""
        return self.nodal_volume() ** (1.0 / max(self.dim, 1))

    def _contour(self, scalar_touched, iso, fields: dict | None, gather: bool) -> Grid:
        names, cols = [], []
        for k, v in (fields or {}).items():
            vt = self.to_touched(v)
            vt = vt.reshape(len(vt), -1)
            names.append((k, vt.shape[1], np.asarray(v).shape[1:]))
            cols.append(vt)
        F = np.hstack(cols) if cols else np.zeros((self._p.n_touched, 0))
        r = self._p.contour_touched(np.ascontiguousarray(scalar_touched, np.float64), float(iso),
                                    np.ascontiguousarray(F), gather)
        cells = Cells.from_dict(r["polys"]).concat(Cells.from_dict(r["lines"]))
        pd, j = {}, 0
        for k, nc, shape in names:
            v = r["fields"][:, j:j + nc]
            pd[k] = v[:, 0] if shape == () else v.reshape((-1,) + shape)
            j += nc
        cd = {"SourceCell": np.concatenate([r["poly_cell"], r["line_cell"]])}
        g = Grid(r["points"], cells, pd, cd)
        g.field_data["edges"] = (r["edge_a"], r["edge_b"], r["t"])
        return g

    def contour(self, scalar, iso: float, fields: dict | None = None, gather: bool = True) -> Grid:
        """Iso-surface scalar == iso of an owned scalar (polygons; lines in 2D),
        with owned `fields` interpolated. With gather=True the result is complete
        on rank 0 and empty elsewhere; with gather=False each rank keeps its piece."""
        return self._contour(self.to_touched(scalar), iso, fields, gather)

    def slice(self, origin, normal, fields: dict | None = None, gather: bool = True) -> Grid:
        """Planar cut through the volume cells."""
        n = np.asarray(normal, np.float64)
        n = n / np.linalg.norm(n)
        d = (self._p.xyz_touched - np.asarray(origin, np.float64)) @ n
        return self._contour(d, 0.0, fields, gather)
