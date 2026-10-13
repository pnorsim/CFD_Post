"""MPI from Python without mpi4py.

MPI is initialised automatically on import when the process was started by an
MPI launcher (``mpirun -n 4 python script.py``); otherwise everything runs as a
serial communicator of size 1 with the same API. Set ``CFDPOST_MPI=1`` to force
initialisation or ``CFDPOST_MPI=0`` to disable it.

Python objects are exchanged pickled; numpy arrays reduce element-wise.
"""
from __future__ import annotations

import atexit
import os
import pickle
import sys

import numpy as np

from . import _core

# Open MPI, PMI/PMI2 (MPICH, Intel MPI, srun --mpi=pmi2), PMIx (srun --mpi=pmix),
# MVAPICH, Hydra, Cray PALS / ALPS
_LAUNCHER_VARS = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMI_RANK", "PMIX_RANK", "MV2_COMM_WORLD_SIZE",
                  "MPI_LOCALNRANKS", "I_MPI_HYDRA_HOST_FILE", "PALS_RANKID", "ALPS_APP_PE")


def launched_by_mpi() -> bool:
    return any(v in os.environ for v in _LAUNCHER_VARS)


def init() -> bool:
    """Initialise MPI if needed. Returns True if MPI is running."""
    if _core.mpi_is_initialized():
        return True
    _core.mpi_init()
    atexit.register(_finalize)
    _refresh()
    return True


def _finalize():
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        _core.mpi_finalize()


class Comm:
    """Communicator. ``Comm.world()`` is MPI_COMM_WORLD when MPI runs, else serial."""

    def __init__(self, core: _core.Comm):
        self._c = core

    @classmethod
    def world(cls) -> "Comm":
        return cls(_core.Comm.world())

    @classmethod
    def self(cls) -> "Comm":
        return cls(_core.Comm.self())

    @property
    def core(self) -> _core.Comm:
        return self._c

    @property
    def rank(self) -> int:
        return self._c.rank

    @property
    def size(self) -> int:
        return self._c.size

    @property
    def is_root(self) -> bool:
        return self._c.rank == 0

    def __repr__(self):
        return f"Comm(rank={self.rank}, size={self.size}, mpi={self._c.is_mpi})"

    def barrier(self):
        self._c.barrier()

    def split(self, color: int, key: int = 0) -> "Comm":
        return Comm(self._c.split(color, key))

    # -------------------------------------------------------------- objects
    def bcast(self, obj=None, root: int = 0):
        if self.size == 1:
            return obj
        data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL) if self.rank == root else b""
        return pickle.loads(self._c.bcast_bytes(data, root))

    def gather(self, obj, root: int = 0):
        """List of objects on root (None elsewhere)."""
        if self.size == 1:
            return [obj]
        out = self._c.gather_bytes(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL), root)
        return [pickle.loads(b) for b in out] if self.rank == root else None

    def allgather(self, obj):
        if self.size == 1:
            return [obj]
        return [pickle.loads(b) for b in self._c.allgather_bytes(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))]

    def send(self, obj, dest: int, tag: int = 0):
        self._c.send_bytes(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL), dest, tag)

    def recv(self, source: int = _core.ANY_SOURCE, tag: int = _core.ANY_TAG, status: bool = False):
        data, src, tg = self._c.recv_bytes(source, tag)
        obj = pickle.loads(data)
        return (obj, src, tg) if status else obj

    # -------------------------------------------------------------- numbers
    def allreduce(self, x, op: str = "sum"):
        """Element-wise reduction of a number or numpy array (op: sum, min, max)."""
        if self.size == 1:
            return x
        scalar = np.ndim(x) == 0
        r = self._c.allreduce(np.asarray(x, dtype=np.float64), op)
        return float(r.reshape(())) if scalar else r

    def sum(self, x):
        return self.allreduce(np.sum(x), "sum")

    def min(self, x):
        return self.allreduce(np.min(x) if np.size(x) else np.inf, "min")

    def max(self, x):
        return self.allreduce(np.max(x) if np.size(x) else -np.inf, "max")

    def count(self, mask) -> int:
        return int(self.allreduce(float(np.count_nonzero(mask)), "sum"))

    def quantile(self, x, q: float) -> float:
        """Exact global q-quantile of values distributed over ranks
        (each value counted once: pass owned arrays). Lower interpolation."""
        x = np.asarray(x, dtype=np.float64).ravel()
        x = x[np.isfinite(x)]
        if self.size == 1:
            return float(np.quantile(x, q, method="lower")) if len(x) else np.nan
        n = self.count(np.ones(len(x), bool))
        if n == 0:
            return np.nan
        k = int(np.floor(q * (n - 1)))   # target position (0-based) in the sorted global array
        xs = np.sort(x)
        count = lambda v: int(self.allreduce(float(np.searchsorted(xs, v, side="right")), "sum"))  # noqa: E731
        lo, hi = self.min(x), self.max(x)
        if count(lo) > k:
            return lo
        c_lo, c_hi = count(lo), n
        # bisection on the value keeping count(lo) <= k < count(hi), until few values remain
        while c_hi - c_lo > 100000:
            mid = 0.5 * (lo + hi)
            if mid <= lo or mid >= hi:
                break
            c = count(mid)
            if c > k:
                hi, c_hi = mid, c
            else:
                lo, c_lo = mid, c
        part = xs[np.searchsorted(xs, lo, side="right"):np.searchsorted(xs, hi, side="right")]
        cand = np.sort(np.concatenate(self.allgather(part)))
        return float(cand[k - c_lo])

    def median(self, x) -> float:
        return self.quantile(x, 0.5)

    def print(self, *args, **kw):
        """print() on rank 0 only (flushes)."""
        if self.rank == 0:
            print(*args, **kw, flush=True)


world: Comm
rank: int
size: int


def _refresh():
    global world, rank, size
    world = Comm.world()
    rank, size = world.rank, world.size


def get_comm(comm=None) -> Comm:
    if comm is None:
        return world
    if isinstance(comm, Comm):
        return comm
    if isinstance(comm, _core.Comm):
        return Comm(comm)
    raise TypeError(f"not a communicator: {comm!r}")


_mode = os.environ.get("CFDPOST_MPI", "auto").lower()
if _mode in ("1", "yes", "true", "on") or (_mode == "auto" and launched_by_mpi()):
    init()
_refresh()
