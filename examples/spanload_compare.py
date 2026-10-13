#!/usr/bin/env python3
"""Span loading of a wing, baseline vs optimised, compared with the elliptic
loading.

    python -m cfdpost.synthetic demo        # artificial case, see cfdpost.synthetic
    python spanload_compare.py demo/wing/wing.su2 demo/wing/base/surface_flow.vtu \
        demo/wing/opt/surface_flow.vtu --volume demo/wing/opt/flow.vtu \
        --markers wing tip --ref-area 21 --farfield-marker farfield

Markers come from the SU2 mesh (both solutions share its topology). Face
orientation comes from the volume elements; the angle of attack from the
far-field velocity of a volume solution (fixed-CL runs).
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

from cfdpost import SurfaceSolution, loads, read_su2

ap = argparse.ArgumentParser()
ap.add_argument("mesh")
ap.add_argument("base")
ap.add_argument("opt")
ap.add_argument("--volume", default=None, help="volume solution for the far-field AoA")
ap.add_argument("--aoa", type=float, default=None)
ap.add_argument("--markers", nargs="+", required=True, help="markers forming the wing")
ap.add_argument("--farfield-marker", default="FAR")
ap.add_argument("--ref-area", type=float, default=1.0)
ap.add_argument("--ref-length", type=float, default=1.0)
ap.add_argument("--nbins", type=int, default=120)
ap.add_argument("-o", "--output", default="spanload_base_opt")
a = ap.parse_args()

mesh = read_su2(a.mesh)                       # elements needed for the face orientation
aoa = a.aoa if a.aoa is not None else loads.freestream_from_volume(a.volume, mesh, marker=a.farfield_marker)[1]
print(f"AoA = {aoa:.4f} deg")

cases = {}
for name, path in (("base", a.base), ("opt", a.opt)):
    wing = SurfaceSolution.read(path, mesh).select(a.markers)
    wing.orientation("mesh", mesh)
    tot = loads.integrate(wing, ref_area=a.ref_area, ref_length=a.ref_length, aoa=aoa)
    cases[name] = (wing, tot)
    c = tot.coefficients()
    print(f"{name}: CL_wing {c['CL']:.5f}  CD_wing {c['CD']:.6f}  (pressure {c['CD_p']:.6f}, friction {c['CD_v']:.6f})")

y = np.concatenate([w.points[:, 1] for w, _ in cases.values()])
edges = np.linspace(y.min(), y.max(), a.nbins + 1)
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
cols, table = {"base": "k", "opt": "tab:red"}, []
gap = np.zeros(a.nbins, bool)
res = {}
for name, (wing, tot) in cases.items():
    sl = loads.spanload(wing, edges=edges, aoa=aoa)
    gap |= sl.open_strips()
    res[name] = sl
semi = edges[-1]
for name, sl in res.items():
    ell, l0 = loads.elliptic(sl, semi, mask=~gap)
    eta = sl.centers / semi
    l = sl.per_span()
    dev = np.where(gap, np.nan, (l - ell) / l0)
    ax[0].plot(eta, l / a.ref_length, cols[name], lw=2, label=name)
    ax[0].plot(eta, ell / a.ref_length, cols[name], ls="--", lw=1, label=f"elliptic ({name})")
    ax[1].plot(eta, 100 * dev, cols[name], lw=2,
               label=f"{name} (RMS {100 * np.sqrt(np.nanmean(dev ** 2)):.2f}%)")
    ax[2].plot(eta, sl.cl(), cols[name], lw=2, label=name)
    table += [l / a.ref_length, ell / a.ref_length, sl.cl()]
for x in ax:
    for k in np.flatnonzero(gap):
        x.axvspan(edges[k] / semi, edges[k + 1] / semi, color="0.9", lw=0)
    x.set_xlabel(r"$\eta = y/s$")
    x.grid(alpha=0.3)
    x.legend(fontsize=8)
ax[0].set_ylabel(r"$c\,c_l/c_{ref}$")
ax[1].set_ylabel("deviation from elliptic [% of root value]")
ax[2].set_ylabel(r"$c_l$")
fig.tight_layout()
fig.savefig(a.output + ".png", dpi=150)
np.savetxt(a.output + ".dat", np.column_stack([res["base"].centers, res["base"].centers / semi, *table]),
           header="y eta  base: c*cl/cref ell cl  opt: c*cl/cref ell cl", fmt="%.6e")
print("wrote", a.output + ".png/.dat")
