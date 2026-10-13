"""Artificial geometries and solutions for tests, examples and demos.

Nothing here comes from a real configuration:

* ``box``           mixed-element box meshes (hex, pyramid, prism, tet; optional jitter
                    and mirrored cells)
* ``icosphere``     closed triangulated sphere
* ``wing_case``     a swept, tapered, twisted NACA 4-digit half wing with a thin
                    prism/hex layer volume mesh, SU2 markers (wing, tip, symmetry,
                    farfield), an analytic transonic-like Cp with an upper-surface
                    shock, and an "optimised" variant (more twist, bending, weaker
                    shock) on the same topology. Written as SU2 meshes and
                    SU2-style surface_flow.vtu / flow.vtu files. The fields are
                    analytic, not a flow solution: fine for exercising the tools,
                    not physically meaningful (e.g. drag).
* ``shock_box``     a box with a curved (paraboloidal) analytic shock and a
                    turbulent "wall" region, written as flow.vtu.

    python -m cfdpost.synthetic outdir      # writes everything below outdir
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .grid import Cells, Grid
from .su2 import write_su2

GAMMA = 1.4

# ---------------------------------------------------------------------------- boxes
# Kuhn subdivision of a hex (VTK node order) into 6 positively oriented tets
_KUHN = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]
_HEX_FACES = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]  # outward
# node permutations mirroring a cell (negative orientation)
_MIRROR = {10: [0, 2, 1, 3], 12: [4, 5, 6, 7, 0, 1, 2, 3], 13: [3, 4, 5, 0, 1, 2], 14: [0, 3, 2, 1, 4]}


def box(n=(6, 5, 4), lo=(0, 0, 0), hi=(1, 1, 1), mix=("hex",), jitter=0.0, flip=0.0, seed=0) -> Grid:
    """Box of n hexes per direction; each hex becomes one of `mix`:
    'hex', 'pyr' (6 pyramids around an added centre), 'wedge' (2 prisms), 'tet' (6 tets).
    jitter moves interior nodes by up to jitter * spacing; a fraction `flip` of
    the cells is written mirrored (negative VTK orientation)."""
    rng = np.random.default_rng(seed)
    nx, ny, nz = n
    xs = [np.linspace(lo[d], hi[d], n[d] + 1) for d in range(3)]
    X, Y, Z = np.meshgrid(*xs, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    if jitter:
        h = (np.asarray(hi) - np.asarray(lo)) / np.asarray(n)
        inner = np.all((pts > np.asarray(lo) + 1e-12) & (pts < np.asarray(hi) - 1e-12), axis=1)
        pts[inner] += jitter * h * rng.uniform(-1, 1, (inner.sum(), 3))
    pid = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k  # noqa: E731
    pts = list(pts)
    types, conn = [], []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                h = [pid(i, j, k), pid(i + 1, j, k), pid(i + 1, j + 1, k), pid(i, j + 1, k),
                     pid(i, j, k + 1), pid(i + 1, j, k + 1), pid(i + 1, j + 1, k + 1), pid(i, j + 1, k + 1)]
                kind = mix[rng.integers(len(mix))]
                if kind == "hex":
                    types.append(12)
                    conn.append(h)
                elif kind == "tet":
                    for t in _KUHN:
                        types.append(10)
                        conn.append([h[a] for a in t])
                elif kind == "wedge":
                    for w in ((0, 2, 1, 4, 6, 5), (0, 3, 2, 4, 7, 6)):
                        types.append(13)
                        conn.append([h[a] for a in w])
                elif kind == "pyr":
                    c = len(pts)
                    pts.append(np.mean([pts[v] for v in h], axis=0))
                    for f in _HEX_FACES:
                        # base normal must point to the apex: reverse the outward face
                        types.append(14)
                        conn.append([h[f[0]], h[f[3]], h[f[2]], h[f[1]], c])
    for i in range(len(conn)):
        if flip and rng.uniform() < flip:
            conn[i] = [conn[i][a] for a in _MIRROR[types[i]]]
    sizes = np.array([len(c) for c in conn])
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    return Grid(np.array(pts), Cells(np.array(types, np.uint8), offsets, np.concatenate(conn)))


def box_boundary(n=(6, 5, 4)):
    """Boundary quads (xmin, xmax, zmin, zmax) of the structured box() numbering,
    right-hand normals pointing out of the box."""
    nx, ny, nz = n
    pid = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k  # noqa: E731
    sides = {
        "xmin": [[pid(0, j, k), pid(0, j, k + 1), pid(0, j + 1, k + 1), pid(0, j + 1, k)]
                 for j in range(ny) for k in range(nz)],
        "xmax": [[pid(nx, j, k), pid(nx, j + 1, k), pid(nx, j + 1, k + 1), pid(nx, j, k + 1)]
                 for j in range(ny) for k in range(nz)],
        "zmin": [[pid(i, j, 0), pid(i, j + 1, 0), pid(i + 1, j + 1, 0), pid(i + 1, j, 0)]
                 for i in range(nx) for j in range(ny)],
        "zmax": [[pid(i, j, nz), pid(i + 1, j, nz), pid(i + 1, j + 1, nz), pid(i, j + 1, nz)]
                 for i in range(nx) for j in range(ny)],
    }
    return {k: np.array(v) for k, v in sides.items()}


def icosphere(subdiv=3, r=1.0):
    """Triangulated sphere (points, triangles) with outward right-hand normals."""
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
         (8, 6, 7), (9, 8, 1)]
    v = [np.array(p, float) / np.linalg.norm(p) for p in v]
    for _ in range(subdiv):
        cache, nf = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = v[a] + v[b]
                v.append(m / np.linalg.norm(m))
                cache[key] = len(v) - 1
            return cache[key]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return np.array(v) * r, np.array(f)


# ---------------------------------------------------------------------------- wing
@dataclass
class WingSpec:
    """Planform and section of the artificial half wing (span along +y, root at y=0)."""
    semispan: float = 10.0
    root_chord: float = 3.0
    taper: float = 0.4
    sweep_le_deg: float = 25.0
    dihedral_deg: float = 4.0
    twist_root_deg: float = 2.0
    twist_tip_deg: float = -2.0
    naca: str = "2412"
    n_chord: int = 41          # points per side (cosine spacing)
    n_span: int = 25
    layers: tuple = (0.01, 0.03, 0.07)   # extrusion distances of the volume layers [m]
    bend: float = 0.0          # tip deflection z = bend * eta^2 [m]

    @property
    def ref_area(self) -> float:
        return 0.5 * self.root_chord * (1 + self.taper) * self.semispan

    def chord(self, eta):
        return self.root_chord * (1 - (1 - self.taper) * eta)

    def twist(self, eta):
        return self.twist_root_deg + (self.twist_tip_deg - self.twist_root_deg) * eta


def _naca4(code: str, x):
    m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    if m == 0:
        yc, dy = np.zeros_like(x), np.zeros_like(x)
    else:
        yc = np.where(x < p, m / p ** 2 * (2 * p * x - x ** 2), m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x - x ** 2))
        dy = np.where(x < p, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    th = np.arctan(dy)
    up = np.stack([x - yt * np.sin(th), yc + yt * np.cos(th)], 1)
    lo = np.stack([x + yt * np.sin(th), yc - yt * np.cos(th)], 1)
    return up, lo


@dataclass
class WingGeometry:
    spec: WingSpec
    points: np.ndarray            # all nodes (surface layer first)
    elements: Cells
    markers: dict
    n_surface: int                # nodes of the wing surface (global ids 0..n_surface-1)
    eta: np.ndarray               # per surface node
    xc: np.ndarray                # chordwise fraction per surface node
    upper: np.ndarray             # bool per surface node
    layer_of: np.ndarray = field(default=None)   # layer index per node
    surface_of: np.ndarray = field(default=None)  # surface node each node was extruded from


def wing_geometry(spec: WingSpec | None = None) -> WingGeometry:
    """Surface + prism/hex layer volume mesh of the half wing."""
    s = spec or WingSpec()
    nc, ns = s.n_chord, s.n_span
    xi = 0.5 * (1 - np.cos(np.linspace(0, np.pi, nc)))          # 0 = LE, 1 = TE
    up, lo = _naca4(s.naca, xi)
    up[-1] = lo[-1] = [1.0, 0.0]                                   # closed trailing edge
    # loop: TE -> lower -> LE -> upper (TE and LE shared)
    loop = np.vstack([lo[::-1], up[1:-1]])
    loop_xc = np.concatenate([xi[::-1], xi[1:-1]])
    loop_up = np.concatenate([np.zeros(nc, bool), np.ones(nc - 2, bool)])
    nl = len(loop)
    eta = np.linspace(0, 1, ns)
    pts = []
    for e in eta:
        c, th = s.chord(e), np.radians(s.twist(e))
        X, Z = loop[:, 0] * c, loop[:, 1] * c
        xq = 0.25 * c                                              # twist about the quarter chord
        Xr = xq + (X - xq) * np.cos(th) + Z * np.sin(th)
        Zr = -(X - xq) * np.sin(th) + Z * np.cos(th)
        y = e * s.semispan
        xle = y * np.tan(np.radians(s.sweep_le_deg))
        zle = y * np.tan(np.radians(s.dihedral_deg)) + s.bend * e ** 2
        pts.append(np.stack([xle + Xr, np.full(nl, y), zle + Zr], 1))
    P = np.vstack(pts)
    nid = lambda j, k: j * nl + (k % nl)  # noqa: E731

    wing = [[nid(j, k), nid(j, k + 1), nid(j + 1, k + 1), nid(j + 1, k)] for j in range(ns - 1) for k in range(nl)]
    # tip cap: ladder between lower and upper points of equal x/c
    L = lambda i: nid(ns - 1, nc - 1 - i)                          # noqa: E731  lower, i = x index
    U = lambda i: nid(ns - 1, nc - 1 + i) if i < nc - 1 else nid(ns - 1, 0)  # noqa: E731
    tip = [[L(0), L(1), U(1)]] + [[L(i), L(i + 1), U(i + 1), U(i)] for i in range(1, nc - 2)] + \
          [[L(nc - 2), L(nc - 1), U(nc - 2)]]
    tip = [f[::-1] for f in tip]                                   # +y normals (out of the wing)
    surf_faces = wing + tip
    tip_check = np.cross(P[tip[1][1]] - P[tip[1][0]], P[tip[1][2]] - P[tip[1][0]])
    assert tip_check[1] > 0

    # node normals: mean of the unit normals of the adjacent faces (not area
    # weighted, so the small tip faces still tilt the tip-edge normals outboard),
    # kept in the symmetry plane at the root
    n = np.zeros_like(P)
    for f in surf_faces:
        q = P[f]
        fn = np.cross(q[2] - q[0], q[-1] - q[1]) if len(f) == 4 else np.cross(q[1] - q[0], q[2] - q[0])
        n[f] += fn / np.linalg.norm(fn)
    root = np.arange(nl)
    n[root, 1] = 0.0
    n /= np.linalg.norm(n, axis=1)[:, None]

    n0 = len(P)
    layers = [P] + [P + d * n for d in s.layers]
    allp = np.vstack(layers)
    nlay = len(s.layers)
    types, conn = [], []
    for l in range(nlay):
        a0, a1 = l * n0, (l + 1) * n0
        for f in surf_faces:
            if len(f) == 3:
                a, b, c = f
                types.append(13)
                conn.append([a0 + a, a0 + c, a0 + b, a1 + a, a1 + c, a1 + b])
            else:
                types.append(12)
                conn.append([a0 + v for v in f] + [a1 + v for v in f])
    sizes = np.array([len(c) for c in conn])
    elements = Cells(np.array(types, np.uint8), np.concatenate([[0], np.cumsum(sizes)]), np.concatenate(conn))

    def faces(lst):
        sz = np.array([len(f) for f in lst])
        return Cells(np.where(sz == 3, 5, 9).astype(np.uint8), np.concatenate([[0], np.cumsum(sz)]),
                     np.concatenate(lst))
    sym = [[l * n0 + root[k], l * n0 + root[(k + 1) % nl], (l + 1) * n0 + root[(k + 1) % nl], (l + 1) * n0 + root[k]]
           for l in range(nlay) for k in range(nl)]
    far = [[nlay * n0 + v for v in f] for f in surf_faces]
    markers = {"wing": faces(wing), "tip": faces(tip), "symmetry": faces(sym), "farfield": faces(far)}
    surf_eta = np.repeat(eta, nl)
    return WingGeometry(s, allp, elements, markers, n0, surf_eta, np.tile(loop_xc, ns), np.tile(loop_up, ns),
                        np.repeat(np.arange(nlay + 1), n0), np.tile(np.arange(n0), nlay + 1))


def wing_cp(g: WingGeometry, shock: float = 0.6, shock_xc: float = 0.55, suction: float = 1.1):
    """Analytic surface Cp: LE stagnation, upper rooftop of level `suction`
    ending in a shock of strength `shock`, scaled spanwise by a near-elliptic factor."""
    A = 0.25 + 0.75 * np.sqrt(np.clip(1 - g.eta ** 2, 0, None))
    xc = g.xc
    st = np.exp(-xc / 0.01)                                        # stagnation at the LE
    jump = 0.5 * (1 + np.tanh((xc - shock_xc) / 0.015)) * (1.0 - 0.6 * (xc - shock_xc))
    cu = -suction * (1 - np.exp(-xc / 0.04)) + 0.35 * xc + shock * jump   # rooftop, recovery
    cl_ = -0.1 + 0.35 * xc
    return st + (1 - st) * A * np.where(g.upper, cu, cl_)


@dataclass
class FlowConditions:
    mach: float = 0.78
    aoa_deg: float = 2.5
    p_inf: float = 26500.0
    T_inf: float = 223.0
    R: float = 287.05

    @property
    def rho_inf(self):
        return self.p_inf / (self.R * self.T_inf)

    @property
    def a_inf(self):
        return np.sqrt(GAMMA * self.p_inf / self.rho_inf)

    @property
    def direction(self):
        a = np.radians(self.aoa_deg)
        return np.array([np.cos(a), 0.0, np.sin(a)])


def wing_fields(g: WingGeometry, fc: FlowConditions, shock: float, suction: float = 1.1) -> dict:
    """SU2-style point arrays on all nodes; Cp decays away from the wall."""
    cp_s = wing_cp(g, shock, suction=suction)
    cp = cp_s[g.surface_of] * np.exp(-1.5 * g.layer_of)
    q = 0.5 * GAMMA * fc.p_inf * fc.mach ** 2
    p = fc.p_inf + q * cp
    rho = fc.rho_inf * (p / fc.p_inf) ** (1 / GAMMA)
    a2 = GAMMA * p / rho
    V2 = (fc.mach * fc.a_inf) ** 2 + 2 / (GAMMA - 1) * (fc.a_inf ** 2 - a2)
    Vmag = np.sqrt(np.clip(V2, 0, None))
    V = Vmag[:, None] * fc.direction[None, :]
    T = p / (rho * fc.R)
    mu = 1.458e-6 * T ** 1.5 / (T + 110.4)
    cf = np.zeros((len(p), 3))
    wall = g.layer_of == 0
    cf[wall, 0] = 0.003 * (1 - 0.5 * g.xc[g.surface_of[wall]])
    return {"Density": rho, "Momentum": rho[:, None] * V, "Pressure": p, "Temperature": T,
            "Mach": Vmag / np.sqrt(a2), "Pressure_Coefficient": cp, "Velocity": V,
            "Laminar_Viscosity": mu, "Skin_Friction_Coefficient": cf,
            "Eddy_Viscosity": mu * np.where(wall, 0.0, 200.0 / (1 + g.layer_of))}


def _write_solution(g: WingGeometry, fields: dict, d: Path):
    d.mkdir(parents=True, exist_ok=True)
    f32 = {k: np.asarray(v, np.float32) for k, v in fields.items()}
    Grid(g.points.astype(np.float32), g.elements, f32).write(d / "flow.vtu")
    # SU2 surface output: nodes of the plotted markers (wing, tip) in increasing global id
    wt = g.markers["wing"].concat(g.markers["tip"])
    ids = np.arange(g.n_surface)
    Grid(g.points[ids].astype(np.float32), wt, {k: v[ids] for k, v in f32.items()}).write(d / "surface_flow.vtu")


@dataclass
class WingCase:
    directory: Path
    spec: WingSpec
    flow: FlowConditions
    base: WingGeometry
    opt: WingGeometry

    @property
    def files(self) -> dict:
        d = self.directory
        return {"mesh": d / "wing.su2", "mesh_opt": d / "wing_opt.su2",
                "base_surface": d / "base" / "surface_flow.vtu", "base_volume": d / "base" / "flow.vtu",
                "opt_surface": d / "opt" / "surface_flow.vtu", "opt_volume": d / "opt" / "flow.vtu"}


def wing_case(directory, spec: WingSpec | None = None, flow: FlowConditions | None = None,
              opt_twist_delta_deg: float = 1.0, opt_bend: float = 0.25) -> WingCase:
    """Write the artificial wing case:

    wing.su2, wing_opt.su2           baseline / optimised mesh (same topology)
    base/flow.vtu, base/surface_flow.vtu, opt/flow.vtu, opt/surface_flow.vtu
    """
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    spec = spec or WingSpec()
    fc = flow or FlowConditions()
    base = wing_geometry(spec)
    ospec = WingSpec(**{**spec.__dict__, "twist_tip_deg": spec.twist_tip_deg + opt_twist_delta_deg,
                        "bend": spec.bend + opt_bend})
    opt = wing_geometry(ospec)
    case = WingCase(d, spec, fc, base, opt)
    f = case.files
    write_su2(f["mesh"], base.points, base.elements, base.markers)
    write_su2(f["mesh_opt"], opt.points, opt.elements, opt.markers)
    _write_solution(base, wing_fields(base, fc, shock=0.6), d / "base")
    # "optimised" variant: weaker shock (the fields are analytic, not a flow solution:
    # loads are self-consistent for testing but not physically meaningful)
    _write_solution(opt, wing_fields(opt, fc, shock=0.15, suction=1.0), d / "opt")
    return case


# ---------------------------------------------------------------------------- shock box
def shock_box(path, n=(30, 16, 16), shock_x=0.8, curvature=0.4, delta=0.06, seed=0) -> Grid:
    """Box [0,2]x[-1,1]^2 with flow along +x through a curved shock
    x = shock_x + curvature (y^2 + z^2). Units with a = 1 (rho = gamma p):
    M 1.45 -> 0.7. A turbulent region (mu_t/mu = 5000) near z = -1 mimics a
    boundary layer for the viscous filter. Written as a .vtu at `path`."""
    g = box(n, lo=(0, -1, -1), hi=(2, 1, 1), mix=("hex", "pyr", "wedge", "tet"), flip=0.2, seed=seed)
    x, y, z = g.points.T
    s = x - (shock_x + curvature * (y ** 2 + z ** 2))
    w = 0.5 * (1 + np.tanh(s / delta))
    u = 1.45 - 0.75 * w
    p = 1.0 + 1.1 * w
    g.point_data.update(Pressure=p, Density=GAMMA * p, Velocity=np.stack([u, 0 * u, 0 * u], 1), Mach=u,
                        Laminar_Viscosity=np.ones_like(u), Eddy_Viscosity=np.where(z < -0.8, 5000.0, 10.0))
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        g.write(path)
    return g


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="write the artificial test cases")
    ap.add_argument("outdir", nargs="?", default="synthetic_case")
    a = ap.parse_args(argv)
    case = wing_case(os.path.join(a.outdir, "wing"))
    shock_box(os.path.join(a.outdir, "shock_box", "flow.vtu"))
    print(f"wing case: {case.directory} (ref area {case.spec.ref_area:.4g} m2, semispan {case.spec.semispan} m, "
          f"AoA {case.flow.aoa_deg} deg, markers {list(case.base.markers)})")
    print(f"shock box: {os.path.join(a.outdir, 'shock_box', 'flow.vtu')}")


if __name__ == "__main__":
    main()
