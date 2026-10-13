"""Run under mpirun: point ids scattered over the whole range (random permutation
plus unreferenced points), as in large SU2 meshes, must give the serial result.
Exercises the sort-based local numbering and the owner -> halo exchange."""
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "python"), str(Path(__file__).resolve().parents[1])]

import numpy as np  # noqa: E402

import meshgen  # noqa: E402
from cfdpost import Volume, mpi  # noqa: E402
from cfdpost.grid import Cells, Grid  # noqa: E402

w = mpi.world
g = meshgen.box((8, 7, 6), lo=(-1, -1, -1), hi=(1, 1, 1), mix=("hex", "pyr", "wedge", "tet"), flip=0.3, seed=3)
rng = np.random.default_rng(0)
n_orphan = 20 * g.n_points
N = g.n_points + n_orphan
new_id = rng.permutation(N)[:g.n_points]            # referenced points spread over [0, N)
pts = rng.normal(size=(N, 3))
pts[new_id] = g.points
f = np.zeros(N)
f[new_id] = np.sin(3 * g.points[:, 0]) + g.points[:, 2] ** 2
r = np.linalg.norm(pts, axis=1)
G = Grid(pts, Cells(g.cells.types, g.cells.offsets, new_id[g.cells.conn]), {"f": f, "r": r})
path = sys.argv[1]
if w.rank == 0:
    G.write(path)
w.barrier()

vol, ser = Volume(path), Volume(path, comm=mpi.Comm.self())
b, e = vol.own_range
np.testing.assert_array_equal(vol.xyz, ser.xyz[b:e])
used = np.zeros(N, bool)
used[new_id] = True
gp, gs = vol.gradient(vol.read("f")), ser.gradient(ser.read("f"))[b:e]
m = used[b:e]
np.testing.assert_allclose(gp[m], gs[m], rtol=1e-12, atol=1e-12)
assert np.all(np.isnan(gp[~m]))                       # orphans have no gradient
np.testing.assert_allclose(vol.read_touched("f"), ser.read("f")[vol.partition.touched])
iso, iso_s = vol.contour(vol.read("r"), 0.7, {"f": vol.read("f")}), ser.contour(ser.read("r"), 0.7, {"f": ser.read("f")})
if w.rank == 0:
    assert iso.n_cells == iso_s.n_cells > 0
    np.testing.assert_allclose(iso.points, iso_s.points, atol=1e-14)
    np.testing.assert_allclose(iso.point_data["f"], iso_s.point_data["f"], atol=1e-13)
w.print("OK", w.size)
