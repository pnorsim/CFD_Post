import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture
def tmp(tmp_path):
    return tmp_path


def have_mpirun():
    return shutil.which("mpirun") is not None



def is_open_mpi() -> bool:
    import subprocess
    p = subprocess.run(["mpirun", "--version"], capture_output=True, text=True)
    return "Open MPI" in p.stdout + p.stderr or "OpenRTE" in p.stdout + p.stderr


def mpirun(n, env_vars=()):
    """Launcher command for n ranks. Open MPI needs --oversubscribe and -x to
    export variables; MPICH/Hydra oversubscribes and forwards the environment."""
    if is_open_mpi():
        cmd = ["mpirun", "--oversubscribe", "-n", str(n)]
        for v in env_vars:
            cmd += ["-x", v]
        return cmd
    return ["mpirun", "-n", str(n)]
