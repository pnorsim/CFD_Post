# Containers

There are three ways to build an image:

| what | command | needs |
|---|---|---|
| Docker image (Open MPI) | `containers/build.sh docker` | Docker |
| Docker image (MPICH ABI) | `containers/build.sh docker mpich` | Docker |
| Singularity/Apptainer image from the Docker image | `containers/build.sh sif [mpich]` | Docker + Singularity/Apptainer |
| Singularity/Apptainer image from source | `containers/build.sh def` | Singularity/Apptainer with `--fakeroot` (or root), no Docker |

The script is a thin wrapper around the commands below. Set `OUT=<dir>` to
choose where `.sif` files go.

The images contain:

- the compiled library and Python package, installed in the system `python3`;
- numpy, scipy, matplotlib and pytest;
- the MPI runtime;
- the tests and examples, in `/opt/cfdpost-src`.

No data are included.

## Docker

```bash
docker build -f containers/Dockerfile -t cfdpost .                       # Open MPI, Ubuntu 24.04
docker build -f containers/Dockerfile --build-arg MPI=mpich \
             --build-arg BASE=debian:bookworm -t cfdpost:mpich .          # MPICH ABI
```

`--build-arg NATIVE=ON` adds `-march=native`. Only use it if the image will
run on the same CPU type it was built on.

The variants use different base images on purpose. Ubuntu 24.04's MPICH
library is built for PMIx, but its `mpiexec` speaks PMI-1, so the ranks start
as independent single processes. Debian 12's MPICH works.

Running it: the entrypoint is `python3 -m cfdpost`. Mount your data at
`/data`, and run as your own user so the files you write belong to you.

```bash
docker run --rm -v $PWD:/data --user $(id -u):$(id -g) cfdpost info flow.vtu
docker run --rm -v $PWD:/data --user $(id -u):$(id -g) cfdpost demo demo          # artificial cases
docker run --rm -v $PWD:/data --user $(id -u):$(id -g) --shm-size=4g \
       --entrypoint mpirun cfdpost -n 8 python3 -m cfdpost shock demo/shock_box/flow.vtu -o shock.vtp
docker run --rm -v $PWD:/data --user $(id -u):$(id -g) --entrypoint python3 cfdpost my_script.py
docker run --rm --entrypoint bash cfdpost -c "cd /opt/cfdpost-src && python3 -m pytest -q"
```

Two settings matter:

- **`--shm-size`:** Open MPI exchanges data between ranks through `/dev/shm`,
  and Docker's 64 MB default is too small for large meshes.
- **Threads:** the image sets `OMP_NUM_THREADS=1`. Use `-e OMP_NUM_THREADS=N`
  for threaded runs without MPI.

## Singularity / Apptainer

### From the Docker image (on a machine with Docker)

```bash
docker save cfdpost:latest -o cfdpost.tar
singularity build cfdpost.sif docker-archive://cfdpost.tar          # or: apptainer build ...
```

Alternatively, push the image to a registry and build from that:
`singularity build cfdpost.sif docker://<registry>/cfdpost:latest`.

Building straight from the local daemon (`docker-daemon://cfdpost:latest`)
fails with some Docker versions (Docker 29 here). The tar archive always works.

### From source, without Docker (e.g. on the cluster)

```bash
cd <repository root>
singularity build --fakeroot cfdpost.sif containers/cfdpost.def
```

`containers/cfdpost.def` installs Open MPI from Ubuntu 24.04. To get the MPICH
ABI, change its `From:` line to `debian:bookworm` and replace the MPI packages
with `mpich libmpich-dev`.

### Checking the image

```bash
singularity test cfdpost.sif                                      # prints version and MPI library
singularity exec --pwd /opt/cfdpost-src cfdpost.sif python3 -m pytest -q
```

## Running Singularity images on a cluster

Singularity mounts `$HOME`, the current directory and `/tmp` automatically.
Add project or scratch filesystems with `--bind /scratch,/work`.

```bash
singularity run cfdpost.sif info flow.vtu                         # = python3 -m cfdpost info ...
singularity exec cfdpost.sif python3 my_script.py
```

### MPI across nodes: hybrid model (recommended)

The host's launcher starts one container process per rank, and MPI inside the
container talks to the host's process manager and network:

```bash
mpirun -n 64 singularity exec cfdpost.sif python3 -m cfdpost shock flow.vtu -o shock.vtp
srun --mpi=pmix singularity exec cfdpost.sif python3 batch_script.py      # Slurm
```

This needs an MPI inside the container that is compatible with the host's.

- **Open MPI hosts:** use the default image, and keep the major version
  close to the host's. The image ships 4.1.6. Tested here: host Open MPI 4.1.6
  launching the container, with identical results to native runs.
- **Cray MPICH, Intel MPI, MVAPICH and other MPICH-ABI hosts:** use the
  `mpich` image.
- **Fast network (InfiniBand/Slingshot):** the in-container MPI only uses it
  if it has the matching libraries. The usual approach is the *bind* model:
  mount the host MPI into the container and put it first on the library path.
  This works because cfdpost only calls the standard MPI C API:

  ```bash
  export SINGULARITYENV_LD_LIBRARY_PATH=/opt/cray/pe/mpich/default/ofi/gnu/9.1/lib:$LD_LIBRARY_PATH
  srun singularity exec --bind /opt/cray,/usr/lib64 cfdpost-mpich.sif python3 script.py
  ```

  Paths are site-specific; your centre's container documentation usually lists
  them.

### Inside one node only

Use the MPI inside the container:

```bash
singularity exec cfdpost.sif mpirun -n 8 python3 -m cfdpost shock flow.vtu -o shock.vtp
```

### Threads

Set threads with `SINGULARITYENV_OMP_NUM_THREADS=8` (or
`APPTAINERENV_OMP_NUM_THREADS=8`), or export `OMP_NUM_THREADS` before
`singularity exec`. The host environment is passed into the container.

Slurm job script for the hybrid model:

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_PROC_BIND=close
srun --mpi=pmix --cpu-bind=cores \
     singularity exec --bind /scratch cfdpost.sif \
     python3 -m cfdpost shock /scratch/case/flow.vtu -o /scratch/case/shock.vtp
```
