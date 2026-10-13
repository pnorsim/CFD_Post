"""cfdpost: CFD post-processing for SU2 meshes and VTK solutions.

C++ (OpenMP + MPI) kernels with a Python interface. Run serial scripts as usual
or the same script with ``mpirun -n N python script.py``.
"""
import os as _os
import sys as _sys

# Open MPI loads its plugins with dlopen and needs the libmpi symbols in the global
# scope; Python loads extensions RTLD_LOCAL. Importing _core with RTLD_GLOBAL puts
# exactly the libmpi it was linked against (not whatever ldconfig finds first, which
# on HPC systems with environment modules may be a different MPI) in global scope.
_flags = _sys.getdlopenflags()
_sys.setdlopenflags(_flags | _os.RTLD_GLOBAL)
try:
    from . import _core  # noqa: E402
finally:
    _sys.setdlopenflags(_flags)

from . import mpi  # noqa: E402
from .grid import Cells, Grid  # noqa: E402
from .io import VtkFile, read_vtk, write_vtk  # noqa: E402
from .su2 import Su2Mesh, read_su2, write_su2, marker_list  # noqa: E402
from .surface import SurfaceSolution  # noqa: E402
from .volume import Volume  # noqa: E402
from . import fields, shock, loads, cuts, deform, batch  # noqa: E402

__version__ = "0.1.0"


def __getattr__(name):
    # lazy: keeps ``python -m cfdpost.synthetic`` free of a double-import warning
    if name == "synthetic":
        import importlib
        return importlib.import_module(".synthetic", __name__)
    raise AttributeError(f"module 'cfdpost' has no attribute {name!r}")

__all__ = ["Cells", "Grid", "VtkFile", "read_vtk", "write_vtk", "Su2Mesh", "read_su2", "write_su2", "marker_list",
           "SurfaceSolution", "Volume", "fields", "shock", "loads", "cuts", "deform", "batch", "synthetic", "mpi"]
