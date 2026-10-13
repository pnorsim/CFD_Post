# cfdpost

Fast **SU2/VTK post-processing** with a Python API and a C++ backend (OpenMP + MPI).

**Features:** Shock detection (Lovely–Haimes and sensors), forces and span loads, surface/volume cuts, SU2 marker extraction, mesh/solution comparisons, and distributed batch processing.

## Install

Requires C++17, CMake ≥ 3.18, MPI, OpenMP, NumPy, SciPy, and pybind11. zlib is optional.

```bash
pip install --no-build-isolation -e .
```

## Quick start

```bash
python -m cfdpost.synthetic demo  # Generate example data
python -m cfdpost info demo/wing/base/flow.vtu demo/wing/wing.su2
python -m cfdpost shock demo/shock_box/flow.vtu -o shock.vtp
python -m cfdpost loads demo/wing/base/surface_flow.vtu --mesh demo/wing/wing.su2 --markers wing tip --ref-area 21 --aoa 2.5
```

Python API:

```python
import cfdpost as cp

vol = cp.Volume("demo/shock_box/flow.vtu")
cp.shock.lovely_haimes(vol, "inclusive").write("shock.vtp")
```

Run CLI commands with MPI using `mpirun -n 8 python -m cfdpost ...`; control threading with `OMP_NUM_THREADS`.

## More

- [Examples](examples/) · [HPC/Slurm](examples/hpc/) · [Docker/Apptainer](containers/README.md)
- Tests: `pytest tests`

**Limits:** ASCII SU2 meshes only; compressed/base64 VTK is decoded per rank; distributed surfaces/slices are gathered on rank 0.