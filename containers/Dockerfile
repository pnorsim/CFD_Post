# cfdpost container: C++ core (OpenMP + MPI) and the Python package.
#
#   docker build -f containers/Dockerfile -t cfdpost .                      # Open MPI (default)
#   docker build -f containers/Dockerfile --build-arg MPI=mpich \
#                --build-arg BASE=debian:bookworm -t cfdpost:mpich .
#
# MPI=mpich gives the MPICH ABI (compatible with Cray MPICH, Intel MPI, MVAPICH)
# for HPC systems where the host MPI is bound into the container. Use it with
# BASE=debian:bookworm: Ubuntu 24.04's libmpich is built for PMIx while its
# mpiexec (Hydra) speaks PMI-1, so ranks start as independent singletons.
# See containers/README.md for running it and for building a Singularity image.

ARG BASE=ubuntu:24.04

# ---------------------------------------------------------------- build stage
FROM ${BASE} AS build
ARG MPI=openmpi
ARG NATIVE=OFF
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ cmake make zlib1g-dev \
        python3 python3-dev python3-venv python3-pip python3-numpy python3-scipy \
        python3-pybind11 python3-setuptools python3-wheel \
        $( [ "$MPI" = "mpich" ] && echo "mpich libmpich-dev" || echo "openmpi-bin libopenmpi-dev" ) \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN rm -rf build python/cfdpost/_core*.so \
    && CMAKE_ARGS="-DCFDPOST_NATIVE=${NATIVE}" python3 -m pip wheel --no-build-isolation --no-deps -w /wheels . \
    && ls -l /wheels

# ---------------------------------------------------------------- runtime stage
FROM ${BASE}
ARG MPI=openmpi
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-numpy python3-scipy python3-matplotlib python3-pytest \
        libgomp1 zlib1g openssh-client \
        $( [ "$MPI" = "mpich" ] && echo "mpich" || echo "openmpi-bin" ) \
    && rm -rf /var/lib/apt/lists/*
# installed into the system python (not a venv): MPI launchers may reset PATH,
# and every rank must find the package with plain `python3`
COPY --from=build /wheels /wheels
RUN python3 -m pip install --break-system-packages --no-deps --no-index /wheels/*.whl && rm -rf /wheels
# tests, examples and docs for in-container checks (python -m pytest /opt/cfdpost-src/tests)
COPY tests /opt/cfdpost-src/tests
COPY examples /opt/cfdpost-src/examples
COPY pyproject.toml /opt/cfdpost-src/
# Open MPI refuses to run as root; work as an unprivileged user
RUN useradd -m -s /bin/bash cfd
USER cfd
WORKDIR /data
ENV OMP_NUM_THREADS=1 \
    MPLBACKEND=Agg \
    OMPI_MCA_btl_vader_single_copy_mechanism=none
LABEL org.opencontainers.image.title="cfdpost" \
      org.opencontainers.image.description="CFD post-processing for SU2 meshes and VTK solutions (C++/OpenMP/MPI + Python)"
# `docker run cfdpost info flow.vtu` / `singularity run cfdpost.sif info flow.vtu` = python3 -m cfdpost ...
# scripts: docker run --entrypoint python3 cfdpost script.py / singularity exec cfdpost.sif python3 script.py
ENTRYPOINT ["python3", "-m", "cfdpost"]
CMD ["--help"]
