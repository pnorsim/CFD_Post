"""Run under mpirun: distributed Volume results must equal the serial ones."""
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "python"), str(Path(__file__).resolve().parents[1])]

import numpy as np  # noqa: E402

import meshgen  # noqa: E402
from cfdpost import Volume, mpi  # noqa: E402
from cfdpost.grid import Grid  # noqa: E402

w = mpi.world
assert w.size > 1, "run with mpirun -n >= 2"
path = sys.argv[1] if len(sys.argv) > 1 else None

g = meshgen.box((9, 8, 7), lo=(-1, -1, -1), hi=(1, 1, 1), mix=("hex", "pyr", "wedge", "tet"), flip=0.3, seed=7)
r = np.linalg.norm(g.points - [0.1, 0, 0], axis=1)
f = np.sin(2 * g.points[:, 0]) + g.points[:, 1] ** 2
g.point_data.update(r=r, f=f)
if path:
    if w.rank == 0:
        g.write(path)
    w.barrier()
    src = path
else:
    src = g

vol = Volume(src)                          # distributed over the world
ser = Volume(src, comm=mpi.Comm.self())    # every rank also computes serially

own = vol.own_range
fd, rd = vol.read("f"), vol.read("r")
np.testing.assert_array_equal(fd, ser.read("f")[own[0]:own[1]])
grad = vol.gradient(fd)
np.testing.assert_allclose(grad, ser.gradient(ser.read("f"))[own[0]:own[1]], rtol=1e-12, atol=1e-12)
np.testing.assert_allclose(vol.nodal_volume(), ser.nodal_volume()[own[0]:own[1]], rtol=1e-12)
np.testing.assert_allclose(vol.xyz, ser.xyz[own[0]:own[1]])

# global statistics on owned arrays
assert abs(w.sum(fd) - ser.read("f").sum()) < 1e-9
for q in (0.0, 0.1, 0.5, 0.93, 1.0):
    assert w.quantile(fd, q) == np.quantile(ser.read("f"), q, method="lower"), q

# iso-surface gathered on root == serial
iso = vol.contour(rd, 0.55, {"f": fd, "g": grad})
iso_s = ser.contour(ser.read("r"), 0.55, {"f": ser.read("f"), "g": ser.gradient(ser.read("f"))})
if w.rank == 0:
    ka, ks = iso.field_data["edges"], iso_s.field_data["edges"]
    np.testing.assert_array_equal(ka[0], ks[0])
    np.testing.assert_array_equal(ka[1], ks[1])
    np.testing.assert_allclose(iso.points, iso_s.points, atol=1e-14)
    np.testing.assert_allclose(iso.point_data["g"], iso_s.point_data["g"], atol=1e-12)
    assert iso.n_cells == iso_s.n_cells
    assert abs(iso.areas().sum() - iso_s.areas().sum()) < 1e-12
else:
    assert iso.n_cells == 0

# slice
s = vol.slice((0, 0, 0.13), (0, 0.3, 1))
s_s = ser.slice((0, 0, 0.13), (0, 0.3, 1))
if w.rank == 0:
    assert abs(s.areas().sum() - s_s.areas().sum()) < 1e-12

# object collectives
assert w.allgather(w.rank) == list(range(w.size))
assert w.bcast({"a": 1} if w.rank == 0 else None) == {"a": 1}
w.print("OK", w.size)
