"""Run under mpirun -n 2: messages above 2 GiB (MPI int counts) in bcast, gather,
allreduce and the halo all-to-all."""
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "python")]

import numpy as np  # noqa: E402

from cfdpost import Volume, mpi  # noqa: E402
from cfdpost.grid import Cells, Grid  # noqa: E402

w = mpi.world
assert w.size == 2
GB = 1 << 30

# bytes: bcast and gather of 2.2 GiB
big = (np.arange((2 * GB + 200 * 2 ** 20) // 8, dtype=np.int64)).tobytes() if w.rank == 0 else b""
got = w.core.bcast_bytes(big, 0)
assert len(got) == 2 * GB + 200 * 2 ** 20
assert np.frombuffer(got[-8:], np.int64)[0] == len(got) // 8 - 1
del big
parts = w.core.gather_bytes(got if w.rank == 1 else b"x", 0)
if w.rank == 0:
    assert len(parts[1]) == len(got) and parts[0] == b"x"
del got, parts

# allreduce of 2.4 GB of doubles
a = np.full(300_000_000, w.rank + 1.0)
s = w.allreduce(a)
assert s[0] == 3.0 and s[-1] == 3.0
del a, s

# halo exchange above 2 GiB: cell i references point N-1-i, so rank 0's cells touch
# only points owned by rank 1 and vice versa
N, K = 1_000_000, 600   # 500k touched points x 600 x 8 B = 2.4 GB per rank
pts = np.zeros((N, 3))
cells = Cells(np.ones(N, np.uint8), np.arange(N + 1), np.arange(N)[::-1].copy())
vol = Volume(Grid(pts, cells))
b, e = vol.own_range
own = np.arange(b, e, dtype=np.float64)[:, None] * 1000 + np.arange(K)
t = vol.to_touched(own)
exp = vol.partition.touched[:, None] * 1000.0 + np.arange(K)
assert t.nbytes > 2 * GB, t.nbytes
assert np.array_equal(t, exp)
back = vol.partition.sum_to_owned(t)
assert np.array_equal(back, own)
w.print("OK", w.size)
