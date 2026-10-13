#!/usr/bin/env python3
"""Lovely-Haimes shock surface of an SU2 volume solution, plus point sensors.

    python shock_surface.py flow.vtu [-o shock.vtp] [--preset inclusive]
    mpirun -n 4 python shock_surface.py flow.vtu        # same result, distributed

    on the artificial curved shock (python -m cfdpost.synthetic demo):
    python shock_surface.py demo/shock_box/flow.vtu --slice-y 0

Writes the filtered surface, the raw Mn=1 iso-surface and a z-slice coloured
with the Ducros sensor and the normalised pressure jump.
"""
import argparse
import time

import numpy as np

from cfdpost import Volume, mpi, shock

ap = argparse.ArgumentParser()
ap.add_argument("volume")
ap.add_argument("-o", "--output", default="shock_surface.vtp")
ap.add_argument("--preset", default="default", choices=list(shock.PRESETS))
ap.add_argument("--slice-y", type=float, default=None, help="also write a y=const slice with sensors")
a = ap.parse_args()

t = time.time()
vol = Volume(a.volume)                      # each rank reads only its cells
res = shock.lovely_haimes(vol, a.preset)
res.write(a.output)
res.write(a.output.replace(".vtp", "_raw.vtp"), raw=True)

if a.slice_y is not None:
    sensors = {"Mn": res.fields["Mn"], "Ducros": shock.ducros(vol),
               "dp_rel": shock.pressure_gradient_sensor(vol), "Mach": vol.read("Mach")}
    s = vol.slice((0, a.slice_y, 0), (0, 1, 0), sensors)
    if vol.comm.is_root:
        s.write(a.output.replace(".vtp", f"_slice_y{a.slice_y:g}.vtp"))

mpi.world.print(f"{vol.n_cells} cells on {mpi.size} rank(s), {time.time() - t:.1f} s; "
                f"|grad p| cutoff {res.params['gradp_cutoff']:.4g}; area "
                f"{res.surface.areas().sum() if vol.comm.is_root else np.nan:.3f} m2")
