"""Command line: python -m cfdpost <command> ...  (prefix with mpirun -n N for MPI)."""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def _info(a):
    from .io import VtkFile
    from .su2 import marker_list
    for path in a.files:
        if path.endswith(".su2"):
            print(path)
            for name, n in marker_list(path).items():
                print(f"  marker {name:24s} {n:>10d} elements")
        else:
            f = VtkFile(path)
            print(f)
            for x in f.arrays():
                print(f"  {x['section']:10s} {x['name']:32s} {x['type']:8s} x{x['ncomp']}  {x['format']}"
                      f"{'' if x['direct'] else ' (decoded)'}")


def _shock(a):
    from . import mpi, shock
    from .volume import Volume
    t = time.time()
    vol = Volume(a.volume)
    mpi.world.print(f"{vol.n_points} points, {vol.n_cells} cells on {mpi.size} rank(s): read {time.time() - t:.1f} s")
    res = shock.lovely_haimes(vol, a.preset, a.mn, a.gradp_factor, a.nut_max)
    res.write(a.output)
    if a.raw:
        res.write(a.raw, raw=True)
    mpi.world.print(f"wrote {a.output} ({time.time() - t:.1f} s)")


def _extract(a):
    from .surface import SurfaceSolution
    sol = SurfaceSolution.from_volume(a.volume, a.mesh, a.markers, arrays=a.arrays)
    sol.write(a.output)
    print(f"{a.output}: {sol.grid.n_cells} faces, {sol.grid.n_points} points")


def _loads(a):
    from . import loads
    from .su2 import read_su2
    from .surface import SurfaceSolution
    mesh = read_su2(a.mesh, elements=a.orient == "mesh")
    sol = SurfaceSolution.read(a.surface, mesh)
    sel = sol.select(a.markers) if a.markers else sol
    sel.orientation(a.orient, mesh if a.orient == "mesh" else None)
    res = loads.integrate(sel, ref_area=a.ref_area, ref_length=a.ref_length, ref_point=a.ref_point,
                          aoa=a.aoa, sideslip=a.sideslip, cf=None if a.inviscid else loads.CF)
    c = res.coefficients()
    print("total  " + "  ".join(f"{k}={c[k]: .6f}" for k in ("CL", "CD", "CSF", "CMx", "CMy", "CMz")))
    for name, r in (res.by_marker or {}).items():
        c = r.coefficients()
        print(f"{name:12s} " + "  ".join(f"{k}={c[k]: .6f}" for k in ("CL", "CD", "CSF", "CMy")))


def _demo(a):
    from .synthetic import main as synth
    synth([a.outdir])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cfdpost", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info", help="arrays of VTK files / markers of SU2 meshes")
    p.add_argument("files", nargs="+")
    p.set_defaults(fn=_info)

    p = sub.add_parser("shock", help="Lovely-Haimes shock surface of a volume solution")
    p.add_argument("volume")
    p.add_argument("-o", "--output", default="shock_surface.vtp")
    p.add_argument("--preset", default="default", choices=["default", "inclusive", "conservative"])
    p.add_argument("--mn", type=float, default=None, help="normal Mach iso level")
    p.add_argument("--gradp-factor", type=float, default=None, help="|grad p| cutoff in medians")
    p.add_argument("--nut-max", type=float, default=None, help="max mu_t/mu")
    p.add_argument("--raw", default=None, help="also write the unfiltered Mn iso-surface here")
    p.set_defaults(fn=_shock)

    p = sub.add_parser("extract", help="markers of an SU2 mesh with data from a volume .vtu")
    p.add_argument("volume")
    p.add_argument("--mesh", required=True)
    p.add_argument("--markers", nargs="+", required=True)
    p.add_argument("--arrays", nargs="*", default=None)
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(fn=_extract)

    p = sub.add_parser("loads", help="integrate Cp/Cf over markers of surface_flow.vtu")
    p.add_argument("surface")
    p.add_argument("--mesh", required=True)
    p.add_argument("--markers", nargs="*", default=None)
    p.add_argument("--ref-area", type=float, default=1.0)
    p.add_argument("--ref-length", type=float, default=1.0)
    p.add_argument("--ref-point", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    p.add_argument("--aoa", type=float, default=0.0)
    p.add_argument("--sideslip", type=float, default=0.0)
    p.add_argument("--orient", default="mesh", choices=["mesh", "closed"])
    p.add_argument("--inviscid", action="store_true")
    p.set_defaults(fn=_loads)

    p = sub.add_parser("demo", help="write the artificial test cases (wing, shock box)")
    p.add_argument("outdir", nargs="?", default="demo")
    p.set_defaults(fn=_demo)

    a = ap.parse_args(argv)
    np.set_printoptions(precision=5)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
