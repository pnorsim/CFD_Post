"""Run under mpirun: batch.pmap with dynamic / static schedules and groups."""
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "python"), str(Path(__file__).resolve().parents[1])]

import numpy as np  # noqa: E402

import meshgen  # noqa: E402
from cfdpost import Volume, batch, mpi  # noqa: E402

w = mpi.world
items = list(range(11))


def square(i):
    return i * i


for sched in ("dynamic", "static"):
    res = batch.pmap(square, items, schedule=sched)
    if w.rank == 0:
        assert res == [i * i for i in items], (sched, res)
    else:
        assert res is None


def area(i, comm):
    """Collective per group: distributed iso-surface area of a sphere of radius r_i."""
    g = meshgen.box((8, 8, 8), lo=(-1, -1, -1), hi=(1, 1, 1), mix=("hex", "pyr"), seed=i)
    vol = Volume(g, comm=comm)
    r = np.linalg.norm(vol.xyz, axis=1)
    iso = vol.contour(r, 0.3 + 0.05 * i)
    return comm.bcast(iso.areas().sum() if comm.rank == 0 else None), comm.size


gs = 2
res = batch.pmap(area, items[:5], group_size=gs)
if w.rank == 0:
    ref = [area(i, mpi.Comm.self())[0] for i in items[:5]]
    assert np.allclose([r[0] for r in res], ref, rtol=1e-12), (res, ref)
    assert all(r[1] == gs for r in res)


def boom(i):
    if i == 3:
        raise ValueError("bad item")
    return i


res = batch.pmap(boom, items, errors="return")
if w.rank == 0:
    assert "bad item" in repr(res[3]) and res[4] == 4
w.print("OK", w.size)
