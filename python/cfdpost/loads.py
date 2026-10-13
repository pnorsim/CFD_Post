"""Aerodynamic loads from surface solutions.

Pressure and friction are integrated exactly for piecewise-linear nodal Cp and
Cf on the surface triangulation (quads split around their centroid):

    dF / q_inf = (-Cp n + Cf) dA,   n = unit normal into the fluid.

Coefficients follow SU2: C_F = F / (q_inf S_ref), C_M = M / (q_inf S_ref L_ref).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _core
from .fields import wind_axes
from .grid import Grid
from .surface import SurfaceSolution

CP = "Pressure_Coefficient"
CF = "Skin_Friction_Coefficient"


def _prep(surf, cp, cf, sign):
    if isinstance(surf, SurfaceSolution):
        g = surf.grid
        if sign is None:
            sign = surf.orientation()
    else:
        g = surf
    cpv = g.point_data[cp] if isinstance(cp, str) else np.asarray(cp)
    if cf is None:
        cfv = None
    else:
        cfv = g.point_data.get(cf) if isinstance(cf, str) else np.asarray(cf)
        if cfv is not None:
            cfv = np.asarray(cfv, np.float64)
            if cfv.ndim == 1:   # 2D solvers may store Cf as a scalar: ignore (no direction)
                raise ValueError("Cf must be a vector field (npts, 3)")
            if cfv.shape[1] == 2:
                cfv = np.hstack([cfv, np.zeros((len(cfv), 1))])
    s = None if sign is None else np.broadcast_to(np.asarray(sign, np.float64), (g.n_cells,)).copy()
    return g, np.asarray(cpv, np.float64), cfv, s


@dataclass
class Loads:
    """Integrated loads (per unit dynamic pressure) and coefficients."""
    force_pressure: np.ndarray
    force_viscous: np.ndarray
    moment: np.ndarray
    area: float
    ref_area: float
    ref_length: float
    axes: dict
    by_marker: dict | None = None

    @property
    def force(self):
        return self.force_pressure + self.force_viscous

    def coefficients(self) -> dict:
        F, S, L = self.force, self.ref_area, self.ref_length
        Fp, Fv = self.force_pressure, self.force_viscous
        d, sd, l = self.axes["drag"], self.axes["side"], self.axes["lift"]
        return {"CL": F @ l / S, "CD": F @ d / S, "CSF": F @ sd / S,
                "CL_p": Fp @ l / S, "CD_p": Fp @ d / S, "CL_v": Fv @ l / S, "CD_v": Fv @ d / S,
                "CFx": F[0] / S, "CFy": F[1] / S, "CFz": F[2] / S,
                "CMx": self.moment[0] / (S * L), "CMy": self.moment[1] / (S * L), "CMz": self.moment[2] / (S * L)}

    def __repr__(self):
        c = self.coefficients()
        return ("Loads(" + ", ".join(f"{k}={c[k]:.5f}" for k in ("CL", "CD", "CSF", "CMx", "CMy", "CMz")) +
                f", area={self.area:.4g})")


def integrate(surf, cp=CP, cf=CF, sign=None, ref_point=(0.0, 0.0, 0.0), ref_area: float = 1.0,
              ref_length: float = 1.0, aoa: float = 0.0, sideslip: float = 0.0, ndim: int = 3,
              by_marker: bool = True) -> Loads:
    """Forces and moments on a surface.

    surf : SurfaceSolution (orientation taken from it) or Grid
    cp, cf: point array names or arrays; cf=None for inviscid / pressure only
    sign : per-face (or global) +1/-1 turning right-hand normals into the fluid
    """
    g, cpv, cfv, s = _prep(surf, cp, cf, sign)
    c = g.cells
    fl = _core.face_loads(g.points, c.types, c.offsets, c.conn, cpv, cfv, s, list(ref_point))
    S, Fp, Fv, M = fl[:, 0:3], fl[:, 3:6], fl[:, 6:9], fl[:, 9:12]
    axes = wind_axes(aoa, sideslip, ndim)
    res = Loads(Fp.sum(0), Fv.sum(0), M.sum(0), float(np.linalg.norm(S, axis=1).sum()), ref_area, ref_length, axes)
    if by_marker and isinstance(surf, SurfaceSolution) and surf.marker_names:
        res.by_marker = {}
        for i, name in enumerate(surf.marker_names):
            m = surf.marker_id == i
            if m.any():
                res.by_marker[name] = Loads(Fp[m].sum(0), Fv[m].sum(0), M[m].sum(0),
                                            float(np.linalg.norm(S[m], axis=1).sum()), ref_area, ref_length, axes)
    return res


def face_loads(surf, cp=CP, cf=CF, sign=None, ref_point=(0.0, 0.0, 0.0)) -> dict:
    """Per-face area vector (into fluid), pressure/viscous force and moment, centroid."""
    g, cpv, cfv, s = _prep(surf, cp, cf, sign)
    c = g.cells
    fl = _core.face_loads(g.points, c.types, c.offsets, c.conn, cpv, cfv, s, list(ref_point))
    return {"S": fl[:, 0:3], "Fp": fl[:, 3:6], "Fv": fl[:, 6:9], "M": fl[:, 9:12], "centroid": fl[:, 12:15]}


@dataclass
class SpanLoad:
    """Loads in strips along a span axis (per unit dynamic pressure)."""
    edges: np.ndarray
    force_pressure: np.ndarray   # (n, 3) per strip
    force_viscous: np.ndarray
    moment: np.ndarray           # (n, 3) about ref_point
    area: np.ndarray             # wetted area per strip
    area_vector: np.ndarray      # (n, 3) sum of S: ~0 for a closed section
    abs_area_vector: np.ndarray  # (n, 3) sum of |S| per component
    chord_min: np.ndarray        # extent of the surface along chord_dir per strip
    chord_max: np.ndarray
    axes: dict
    ref_point: np.ndarray

    @property
    def centers(self):
        return 0.5 * (self.edges[1:] + self.edges[:-1])

    @property
    def widths(self):
        return np.diff(self.edges)

    @property
    def force(self):
        return self.force_pressure + self.force_viscous

    @property
    def chord(self):
        return self.chord_max - self.chord_min

    def per_span(self, direction="lift") -> np.ndarray:
        """Force per unit span along a wind axis name or a vector: e.g. c*cl [m]."""
        d = self.axes[direction] if isinstance(direction, str) else np.asarray(direction, float)
        return (self.force @ d) / self.widths

    def cl(self, direction="lift") -> np.ndarray:
        """Sectional coefficient: per-span force / local chord."""
        return self.per_span(direction) / self.chord

    def open_strips(self, tol: float = 0.03, component: int = 2) -> np.ndarray:
        """Strips whose surface does not close in `component` (e.g. z): pylon
        cut-outs, fairing junctions, where the selected markers miss part of the section."""
        up = 0.5 * (self.abs_area_vector[:, component] + self.area_vector[:, component])
        lo = 0.5 * (self.abs_area_vector[:, component] - self.area_vector[:, component])
        return np.abs(up - lo) / np.maximum(np.maximum(up, lo), 1e-300) > tol

    def total(self) -> np.ndarray:
        return self.force.sum(0)

    def to_table(self, direction="lift", ref_length: float = 1.0) -> np.ndarray:
        """Columns: center, eta (center / max edge), c*cl, c*cl/ref_length, chord, cl."""
        y = self.centers
        l = self.per_span(direction)
        return np.column_stack([y, y / self.edges[-1], l, l / ref_length, self.chord, l / self.chord])


def spanload(surf, axis=(0.0, 1.0, 0.0), nbins: int = 100, edges=None, cp=CP, cf=CF, sign=None,
             aoa: float = 0.0, sideslip: float = 0.0, chord_dir=(1.0, 0.0, 0.0), ref_point=None,
             span_range=None, ndim: int = 3) -> SpanLoad:
    """Strip loads along `axis`. Faces are clipped exactly at strip boundaries.

    edges      : strip boundaries along axis (default: nbins equal strips over span_range,
                 which defaults to the extent of the surface)
    chord_dir  : direction used for the local chord (extent per strip)
    ref_point  : moment reference (default origin)
    """
    g, cpv, cfv, s = _prep(surf, cp, cf, sign)
    ax = np.asarray(axis, float)
    ax = ax / np.linalg.norm(ax)
    if edges is None:
        if span_range is None:
            sp = g.points @ ax
            span_range = (sp.min(), sp.max())
        edges = np.linspace(span_range[0], span_range[1], nbins + 1)
    edges = np.asarray(edges, float)
    ref = np.zeros(3) if ref_point is None else np.asarray(ref_point, float)
    c = g.cells
    r = _core.strip_loads(g.points, c.types, c.offsets, c.conn, cpv, cfv, s, list(ref), list(ax),
                          list(edges), list(np.asarray(chord_dir, float)))
    return SpanLoad(edges, r[:, 0:3], r[:, 3:6], r[:, 6:9], r[:, 9], r[:, 10:13], r[:, 13:16], r[:, 16], r[:, 17],
                    wind_axes(aoa, sideslip, ndim), ref)


def elliptic(sl: SpanLoad, semispan: float | None = None, direction="lift", mask=None):
    """Elliptic loading with the same integrated load as `sl` over its strips,
    with the ellipse spanning [0, semispan] from the symmetry plane.

    Returns (per-span load of the ellipse at strip centres, l0 = root value)."""
    s = sl.edges[-1] if semispan is None else semispan
    y = sl.centers
    shape = np.sqrt(np.clip(1.0 - (y / s) ** 2, 0.0, None))
    l = sl.per_span(direction)
    w = sl.widths
    m = np.ones(len(y), bool) if mask is None else np.asarray(mask)
    l0 = np.sum((l * w)[m]) / np.sum((shape * w)[m])
    return l0 * shape, l0


def freestream_from_volume(vtu, mesh=None, marker: str = "FAR", fraction: float = 0.98,
                           velocity: str = "Velocity"):
    """Free-stream velocity and angles from a volume solution: mean velocity on
    the far-field marker nodes (if an Su2Mesh is given) or on the points beyond
    `fraction` of the largest distance from the domain centre.
    Returns (U, aoa_deg, sideslip_deg)."""
    from .io import VtkFile
    f = vtu if isinstance(vtu, VtkFile) else VtkFile(vtu)
    if mesh is not None and marker in mesh.markers:
        ids = np.unique(mesh.markers[marker].conn)
    else:
        P = f.points(dtype=np.float32)
        r = np.linalg.norm(P - P.mean(0), axis=1)
        ids = np.flatnonzero(r > fraction * r.max())
        del P
    U = f.point(velocity, ids=ids).mean(0)
    aoa = np.degrees(np.arctan2(U[2], U[0]))
    beta = np.degrees(np.arcsin(U[1] / np.linalg.norm(U)))
    return U, float(aoa), float(beta)
