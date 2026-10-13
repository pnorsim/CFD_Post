"""Planar cuts of surfaces (sections) and volumes (slices)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import _core
from .grid import Cells, Grid
from .surface import SurfaceSolution


def _grid(surf) -> Grid:
    return surf.grid if isinstance(surf, SurfaceSolution) else surf


def cut_surface(surf, origin, normal, fields=None) -> Grid:
    """Intersect a surface with the plane through `origin` with `normal`.
    Returns line segments with point data interpolated (all arrays by default)."""
    g = _grid(surf)
    n = np.asarray(normal, float)
    n /= np.linalg.norm(n)
    d = (g.points - np.asarray(origin, float)) @ n
    names = list(g.point_data) if fields is None else list(fields)
    names = [k for k in names if np.issubdtype(np.asarray(g.point_data[k]).dtype, np.number)]
    cols, meta = [], []
    for k in names:
        v = np.asarray(g.point_data[k], np.float64)
        cols.append(v.reshape(len(v), -1))
        meta.append((k, cols[-1].shape[1], v.shape[1:]))
    F = np.hstack(cols) if cols else np.zeros((g.n_points, 0))
    c = g.cells
    r = _core.contour(g.points, c.types, c.offsets, c.conn, d, 0.0, np.ascontiguousarray(F))
    pd, j = {}, 0
    for k, nc, shape in meta:
        v = r["fields"][:, j:j + nc]
        pd[k] = v[:, 0] if shape == () else v.reshape((-1,) + shape)
        j += nc
    cells = Cells.from_dict(r["lines"]).concat(Cells.from_dict(r["polys"]))
    cell_data = {"SourceCell": np.concatenate([r["line_cell"], r["poly_cell"]])}
    if isinstance(surf, SurfaceSolution) and len(surf.marker_id):
        cell_data["MarkerID"] = surf.marker_id[cell_data["SourceCell"]]
    out = Grid(r["points"], cells, pd, cell_data)
    out.field_data["plane"] = (np.asarray(origin, float), n)
    return out


def chain_segments(lines: Grid) -> list[np.ndarray]:
    """Order line segments into polylines. Returns point-index arrays; closed
    loops repeat their first index at the end."""
    seg = lines.cells.conn.reshape(-1, 2)
    if len(seg) == 0:
        return []
    n = lines.n_points
    # adjacency (each point has <= 2 segments in a manifold cut)
    nbr = [[] for _ in range(n)]
    for s, (a, b) in enumerate(seg):
        if a == b:
            continue
        nbr[a].append(s)
        nbr[b].append(s)
    used = np.zeros(len(seg), bool)
    chains = []
    # start from open ends first, then remaining loops
    starts = [p for p in range(n) if len(nbr[p]) == 1] + list(range(n))
    for p0 in starts:
        if not any(not used[s] for s in nbr[p0]):
            continue
        chain = [p0]
        p = p0
        while True:
            nxt = [s for s in nbr[p] if not used[s]]
            if not nxt:
                break
            s = nxt[0]
            used[s] = True
            a, b = seg[s]
            p = b if a == p else a
            chain.append(p)
            if p == p0:
                break
        if len(chain) > 1:
            chains.append(np.array(chain))
    return chains


@dataclass
class Section:
    """An ordered cut polyline with point data."""
    points: np.ndarray
    data: dict = field(default_factory=dict)
    closed: bool = False

    def __len__(self):
        return len(self.points)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())

    def trailing_edge(self, chord_dir=(1, 0, 0)) -> int:
        return int(np.argmax(self.points @ np.asarray(chord_dir, float)))

    def leading_edge(self, chord_dir=(1, 0, 0)) -> int:
        """Point farthest from the trailing edge."""
        te = self.points[self.trailing_edge(chord_dir)]
        return int(np.argmax(np.linalg.norm(self.points - te, axis=1)))

    def chord_line(self, chord_dir=(1, 0, 0)):
        """(LE point, TE point, chord length). TE = midpoint of the most aft points
        of upper and lower sides for blunt trailing edges is not attempted: the
        single most aft point is used."""
        le = self.points[self.leading_edge(chord_dir)]
        te = self.points[self.trailing_edge(chord_dir)]
        return le, te, float(np.linalg.norm(te - le))

    def twist(self, chord_dir=(1, 0, 0), up=(0, 0, 1)) -> float:
        """Geometric twist [deg] of the chord line: positive nose-up."""
        le, te, _ = self.chord_line(chord_dir)
        c = te - le
        return float(np.degrees(np.arctan2(-(c @ np.asarray(up, float)), c @ np.asarray(chord_dir, float))))

    def normalized(self, chord_dir=(1, 0, 0), up=(0, 0, 1), frame=None):
        """(x/c, z/c) in the chord frame (LE at origin, chord along x/c, no twist
        removal unless frame='chord'). frame: None -> axes chord_dir/up with LE
        origin and chord normalisation; 'chord' -> rotated into the chord line."""
        le, te, c = self.chord_line(chord_dir)
        d = self.points - le
        if frame == "chord":
            ex = (te - le) / c
            upv = np.asarray(up, float)
            ez = upv - (upv @ ex) * ex
            ez /= np.linalg.norm(ez)
            return d @ ex / c, d @ ez / c
        return d @ np.asarray(chord_dir, float) / c, d @ np.asarray(up, float) / c

    def split(self, chord_dir=(1, 0, 0), up=(0, 0, 1)):
        """Split a closed section at LE and TE into (upper, lower) index arrays,
        each ordered from LE to TE."""
        n = len(self.points) - (1 if self.closed else 0)
        le, te = self.leading_edge(chord_dir) % n, self.trailing_edge(chord_dir) % n
        a = np.arange(le, le + ((te - le) % n) + 1) % n          # LE -> TE forward
        b = (np.arange(te, te + ((le - te) % n) + 1) % n)[::-1]  # LE -> TE backward
        za = self.points[a] @ np.asarray(up, float)
        zb = self.points[b] @ np.asarray(up, float)
        return (a, b) if za.mean() >= zb.mean() else (b, a)


def sections(surf, origin, normal, fields=None, min_points: int = 3) -> list[Section]:
    """All ordered polylines of a planar cut, longest first."""
    cut = cut_surface(surf, origin, normal, fields)
    out = []
    for ch in chain_segments(cut):
        if len(ch) < min_points:
            continue
        closed = ch[0] == ch[-1]
        out.append(Section(cut.points[ch], {k: v[ch] for k, v in cut.point_data.items()}, bool(closed)))
    out.sort(key=lambda s: -s.length)
    return out


def section_at(surf, origin, normal, seed=None, fields=None) -> Section:
    """The section closest to `seed` (default: the longest one), e.g. the wing
    loop among nacelle / fuselage / tail pieces of the same cut."""
    secs = sections(surf, origin, normal, fields)
    if not secs:
        raise ValueError(f"empty cut at origin {origin}")
    if seed is None:
        return secs[0]
    seed = np.asarray(seed, float)
    return min(secs, key=lambda s: np.min(np.linalg.norm(s.points - seed, axis=1)))


def slice_volume(vol, origin, normal, fields: dict | None = None, gather: bool = True) -> Grid:
    """Planar slice through a Volume (see Volume.slice). `fields` maps names to
    owned arrays; strings are read from the file."""
    if isinstance(fields, (list, tuple)):
        f = {k: vol.read(k) for k in fields}
    else:
        f = {k: vol.read(v) if isinstance(v, str) else v for k, v in (fields or {}).items()}
    return vol.slice(origin, normal, f, gather)
