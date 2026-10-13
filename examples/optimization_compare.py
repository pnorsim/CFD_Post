#!/usr/bin/env python3
"""Baseline vs optimised geometry and solution on the same mesh topology.

    python optimization_compare.py BASE.su2 DEFORMED.su2 base/surface_flow.vtu opt/surface_flow.vtu \
        [--markers ...]

    on the artificial case (python -m cfdpost.synthetic demo):
    python optimization_compare.py demo/wing/wing.su2 demo/wing/wing_opt.su2 \
        demo/wing/base/surface_flow.vtu demo/wing/opt/surface_flow.vtu --markers wing tip

Writes (for ParaView):
  shape_change.vtu   markers on the deformed geometry: Displacement, NormalDisplacement
  solution_delta.vtu Cp/Mach of both solutions and their difference, point by point
The deformed mesh may also be given as a volume flow.vtu (points in global order).
"""
import argparse

import numpy as np

from cfdpost import SurfaceSolution, deform, read_su2

ap = argparse.ArgumentParser()
ap.add_argument("base_mesh")
ap.add_argument("deformed")
ap.add_argument("base_surface")
ap.add_argument("opt_surface")
ap.add_argument("--markers", nargs="+", default=None)
a = ap.parse_args()

base = read_su2(a.base_mesh, elements=False)
pair = deform.MeshPair(base, a.deformed)
print("displacement:", {k: round(v, 5) if isinstance(v, float) else v for k, v in pair.summary(a.markers).items()})
pair.surface(a.markers).write("shape_change.vtu")

fields = ["Pressure_Coefficient", "Mach"]
s0 = SurfaceSolution.read(a.base_surface, base, arrays=fields)
s1 = SurfaceSolution.read(a.opt_surface, base, arrays=fields)   # tagged by topology
cmp = deform.compare(s0, s1)
if a.markers:
    cmp = cmp.select(a.markers)
cmp.write("solution_delta.vtu")
d = cmp.grid.point_data["delta_Pressure_Coefficient"]
print(f"delta Cp: min {d.min():.3f} max {d.max():.3f} rms {np.sqrt(np.mean(d ** 2)):.4f}")
