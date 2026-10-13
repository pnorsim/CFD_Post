"""Process many solutions over MPI ranks.

    from cfdpost import batch
    def work(path):
        return shock.lovely_haimes(Volume(path, comm=mpi.Comm.self())).params
    results = batch.pmap(work, files)        # list on rank 0, None elsewhere

    mpirun -n 24 python script.py

With ``group_size=k`` the ranks form groups of k and each item is processed
collectively by a group: ``fn(item, comm)`` receives the group communicator,
e.g. to open a huge volume as a distributed ``Volume(path, comm)``. Set
OMP_NUM_THREADS so that ranks x threads matches the cores.
"""
from __future__ import annotations

import inspect
import traceback

from .mpi import Comm, get_comm

_TAG_READY, _TAG_WORK = 101, 102


class _Failure:
    def __init__(self, index, item, exc):
        self.index, self.item = index, item
        self.message = f"{type(exc).__name__}: {exc}"
        self.traceback = traceback.format_exc()

    def __repr__(self):
        return f"<failed item {self.index}: {self.message}>"


def _call(fn, item, comm, wants_comm):
    return fn(item, comm) if wants_comm else fn(item)


def _wants_comm(fn) -> bool:
    try:
        params = [p for p in inspect.signature(fn).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)]
    except (TypeError, ValueError):
        return False
    return len(params) >= 2 or any(p.kind == p.VAR_POSITIONAL for p in params)


def pmap(fn, items, comm=None, group_size: int = 1, schedule: str = "dynamic", errors: str = "raise",
         verbose: bool = False):
    """Map fn over items using all ranks of `comm`.

    fn        : fn(item) or fn(item, comm) (the communicator of the group processing it)
    schedule  : 'dynamic' (rank 0 hands out items as ranks become free; best for
                files of different size) or 'static' (round robin)
    errors    : 'raise' re-raises the first failure on rank 0 after all items ran;
                'return' puts failure objects in the result list
    Returns the list of results in item order on rank 0, None on other ranks.
    """
    world: Comm = get_comm(comm)
    items = list(items)
    wants = _wants_comm(fn)
    if world.size == 1:
        out = []
        for i, it in enumerate(items):
            try:
                out.append(_call(fn, it, Comm.self(), wants))
            except Exception as e:  # noqa: BLE001
                if errors == "raise":
                    raise
                out.append(_Failure(i, it, e))
        return out

    dynamic = schedule == "dynamic" and (world.size - 1) // group_size >= 2
    if dynamic:
        results = _dynamic(fn, items, world, group_size, wants, verbose)
    else:
        results = _static(fn, items, world, group_size, wants, verbose)
    if world.rank == 0 and errors == "raise":
        for r in results:
            if isinstance(r, _Failure):
                raise RuntimeError(f"item {r.index} ({r.item!r}) failed:\n{r.traceback}")
    return results


def _groups(world: Comm, first: int, group_size: int):
    """Split ranks >= first into groups; returns (group comm or None, group index, ngroups)."""
    workers = world.size - first
    ngroups = max(1, workers // group_size)
    r = world.rank - first
    color = min(r // group_size, ngroups - 1) if r >= 0 else -1
    g = world.split(color if color >= 0 else 10 ** 6, world.rank)
    return (g if color >= 0 else None), color, ngroups


def _run(fn, i, item, group, wants, verbose):
    try:
        res = _call(fn, item, group, wants)
    except Exception as e:  # noqa: BLE001
        res = _Failure(i, item, e)
    if verbose and group.rank == 0:
        print(f"[cfdpost.batch] item {i} done", flush=True)
    return res


def _static(fn, items, world, group_size, wants, verbose):
    group, gi, ng = _groups(world, 0, group_size)
    mine = {}
    for i in range(gi, len(items), ng):
        res = _run(fn, i, items[i], group, wants, verbose)
        if group.rank == 0:
            mine[i] = res
    allres = world.gather(mine, 0)
    if world.rank != 0:
        return None
    out = [None] * len(items)
    for d in allres:
        for i, r in d.items():
            out[i] = r
    return out


def _dynamic(fn, items, world, group_size, wants, verbose):
    group, gi, ng = _groups(world, 1, group_size)
    if world.rank == 0:                      # dispatcher
        out = [None] * len(items)
        nxt, active = 0, ng
        while active:
            (idx, res), src, _ = world.recv(tag=_TAG_READY, status=True)
            if idx is not None:
                out[idx] = res
            if nxt < len(items):
                world.send(nxt, src, _TAG_WORK)
                nxt += 1
            else:
                world.send(None, src, _TAG_WORK)
                active -= 1
        return out
    leader = group.rank == 0
    last = (None, None)
    while True:
        if leader:
            world.send(last, 0, _TAG_READY)
            idx = world.recv(0, _TAG_WORK)
        else:
            idx = None
        idx = group.bcast(idx, 0)
        if idx is None:
            return None
        res = _run(fn, idx, items[idx], group, wants, verbose)
        last = (idx, res)
