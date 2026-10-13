"""Test helpers: re-exported from cfdpost.synthetic, plus a small SU2 writer
taking (n, k) marker node arrays."""
import numpy as np

from cfdpost.grid import Cells, Grid
from cfdpost.su2 import write_su2 as _write_su2
from cfdpost.synthetic import box, box_boundary, icosphere  # noqa: F401


def write_su2(path, grid: Grid, markers: dict, index_column=True):
    mk = {k: Cells.uniform(5 if np.shape(v)[1] == 3 else 9, v) for k, v in markers.items()}
    _write_su2(path, grid.points, grid.cells, mk, 3, index_column)
