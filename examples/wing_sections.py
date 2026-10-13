#!/usr/bin/env python3
"""Wing sections and Cp at spanwise stations for several solutions.

    python wing_sections.py base/surface_flow.vtu opt/surface_flow.vtu \
        --semispan S --eta 0.15 0.3 0.45 0.6 0.75 0.9 [--x-max X]

    on the artificial case (python -m cfdpost.synthetic demo):
    python wing_sections.py demo/wing/base/surface_flow.vtu demo/wing/opt/surface_flow.vtu --semispan 10

Each cut is chained into ordered loops; the wing loop is the one holding the
most aft point forward of --x-max (tail planes lie behind it).
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from cfdpost import SurfaceSolution, cuts

ap = argparse.ArgumentParser()
ap.add_argument("surfaces", nargs="+")
ap.add_argument("--semispan", type=float, required=True)
ap.add_argument("--eta", type=float, nargs="+", default=[0.15, 0.3, 0.45, 0.6, 0.75, 0.9])
ap.add_argument("--x-max", type=float, default=np.inf)
ap.add_argument("-o", "--output", default="sections")
a = ap.parse_args()

sols = [SurfaceSolution.read(p, arrays=["Pressure_Coefficient"]) for p in a.surfaces]
n = len(a.eta)
fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
for i, eta in enumerate(a.eta):
    y0 = eta * a.semispan
    for k, (path, sol) in enumerate(zip(a.surfaces, sols)):
        loops = [s for s in cuts.sections(sol, (0, y0, 0), (0, 1, 0)) if s.points[:, 0].min() < a.x_max]
        wing = max(loops, key=lambda s: s.points[s.points[:, 0] < a.x_max, 0].max())
        xc, zc = wing.normalized()
        up, lo = wing.split()
        lab = Path(path).parent.name or path
        for idx in (up, lo):
            axes[0, i].plot(xc[idx], wing.data["Pressure_Coefficient"][idx], f"C{k}", lw=1,
                            label=lab if idx is up else None)
        axes[1, i].plot(xc, zc, f"C{k}", lw=0.8)
        print(f"eta {eta:.2f} {lab}: chord {wing.chord_line()[2]:.4f} m, twist {wing.twist():+.3f} deg")
    axes[0, i].invert_yaxis()
    axes[0, i].set_title(f"$\\eta$ = {eta:.2f}")
    axes[1, i].set_aspect("equal")
    axes[1, i].set_xlabel("x/c")
axes[0, 0].set_ylabel("$C_p$")
axes[0, 0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(a.output + ".png", dpi=150)
print("wrote", a.output + ".png")
