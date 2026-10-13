#!/usr/bin/env python3
"""Post-process many SU2 cases in parallel: one shock surface and the wing
loads per case, collected into a table on rank 0.

    mpirun -n 8 python batch_cases.py MESH.su2 case1/ case2/ ... --markers wing tip   (OMP_NUM_THREADS=3)
    mpirun -n 8 python batch_cases.py MESH.su2 case*/ --markers ... --group 4       (4 ranks per huge case)

    on the artificial case:  python -m cfdpost.synthetic demo
    mpirun -n 3 python batch_cases.py demo/wing/wing.su2 demo/wing/base demo/wing/opt \
        --markers wing tip --ref-area 21 --farfield-marker farfield

Each case directory holds flow.vtu and surface_flow.vtu computed on meshes with
the topology of MESH.su2 (e.g. the design iterations of an optimisation).
Shock surfaces are written to --outdir as <case>_shock.vtp.
"""
import argparse
from pathlib import Path

from cfdpost import SurfaceSolution, Volume, batch, loads, mpi, read_su2, shock

ap = argparse.ArgumentParser()
ap.add_argument("mesh")
ap.add_argument("cases", nargs="+")
ap.add_argument("--markers", nargs="+", required=True, help="markers whose loads are integrated")
ap.add_argument("--farfield-marker", default="FAR")
ap.add_argument("--ref-area", type=float, default=1.0)
ap.add_argument("--group", type=int, default=1, help="ranks per case")
ap.add_argument("--outdir", default=".", help="where <case>_shock.vtp files are written")
a = ap.parse_args()

# marker connectivity only (no points, no elements): O(boundary) memory on every rank;
# tagging surface solutions is topological and does not need coordinates
mesh = read_su2(a.mesh, elements=False, points=False)


def process(case, comm):
    d = Path(case)
    res = shock.lovely_haimes(Volume(d / "flow.vtu", comm), "default", verbose=False)
    res.write(Path(a.outdir) / f"{d.name}_shock.vtp", comm=comm)
    out = {"case": d.name, "shock_area": None}
    if comm.is_root:
        out["shock_area"] = float(res.surface.areas().sum())
        _, aoa, _ = loads.freestream_from_volume(d / "flow.vtu", mesh, marker=a.farfield_marker)
        wing = SurfaceSolution.read(d / "surface_flow.vtu", mesh).select(a.markers)
        wing.orientation("closed")
        c = loads.integrate(wing, ref_area=a.ref_area, aoa=aoa).coefficients()
        out.update(aoa=aoa, CL_wing=c["CL"], CD_wing=c["CD"])
    return out


rows = batch.pmap(process, a.cases, group_size=a.group, verbose=True)
if mpi.rank == 0:
    for r in rows:
        print("  ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
