import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import have_mpirun, is_open_mpi, mpirun

HERE = Path(__file__).parent
pytestmark = [pytest.mark.mpi, pytest.mark.skipif(not have_mpirun(), reason="mpirun not found")]


def run(script, n, *args, mpi_args=()):
    env = dict(os.environ, OMP_NUM_THREADS="2")
    launcher = mpirun(n)
    cmd = [*launcher[:1], *mpi_args, *launcher[1:], sys.executable, str(HERE / "mpi_scripts" / script), *args]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    assert p.returncode == 0, p.stdout + p.stderr
    assert f"OK {n}" in p.stdout, p.stdout + p.stderr


@pytest.mark.parametrize("n", [2, 3])
def test_partition_matches_serial_in_memory(n):
    run("check_partition.py", n)


def test_partition_matches_serial_from_file(tmp_path):
    run("check_partition.py", 4, str(tmp_path / "part.vtu"))


def test_batch(tmp_path):
    run("check_batch.py", 5)


def test_scattered_point_ids(tmp_path):
    run("check_scattered.py", 4, str(tmp_path / "scattered.vtu"))


def test_tcp_transport_like_multinode(tmp_path):
    """Disable shared memory so every message goes through the network stack, as
    between nodes (Open MPI only)."""
    if not is_open_mpi():
        pytest.skip("Open MPI specific option")
    run("check_partition.py", 3, str(tmp_path / "tcp.vtu"), mpi_args=["--mca", "btl", "tcp,self", "--mca", "btl_tcp_if_include", "lo"])
    run("check_scattered.py", 3, str(tmp_path / "tcp2.vtu"),
        mpi_args=["--mca", "btl", "tcp,self", "--mca", "btl_tcp_if_include", "lo"])


@pytest.mark.slow
def test_messages_above_2GiB():
    """~25 GB of RAM; run with: pytest -m slow"""
    run("check_large.py", 2)
