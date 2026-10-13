#!/usr/bin/env bash
# Build cfdpost container images.
#
#   containers/build.sh docker [openmpi|mpich]   Docker image cfdpost:latest / cfdpost:mpich
#   containers/build.sh sif    [openmpi|mpich]   Docker image, then converted to cfdpost[-mpich].sif
#   containers/build.sh def                      cfdpost.sif from containers/cfdpost.def (no Docker,
#                                                needs --fakeroot or root; e.g. on a cluster login node)
#
# Environment: SINGULARITY (singularity or apptainer, auto-detected), OUT (output directory, default .)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-docker}"
MPI="${2:-openmpi}"
OUT="${OUT:-$PWD}"
SING="${SINGULARITY:-$(command -v apptainer || command -v singularity || true)}"

if [[ "$MPI" == "mpich" ]]; then
  TAG=cfdpost:mpich; SIF=cfdpost-mpich.sif
  ARGS=(--build-arg MPI=mpich --build-arg BASE=debian:bookworm)
else
  TAG=cfdpost:latest; SIF=cfdpost.sif
  ARGS=()
fi

build_docker() {
  docker build -f "$ROOT/containers/Dockerfile" "${ARGS[@]}" -t "$TAG" "$ROOT"
}

case "$MODE" in
  docker)
    build_docker
    ;;
  sif)
    [[ -n "$SING" ]] || { echo "singularity/apptainer not found" >&2; exit 1; }
    build_docker
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    # docker-archive is the portable route (docker-daemon:// fails with some Docker versions)
    docker save "$TAG" -o "$tmp/image.tar"
    "$SING" build --force "$OUT/$SIF" "docker-archive://$tmp/image.tar"
    ;;
  def)
    [[ -n "$SING" ]] || { echo "singularity/apptainer not found" >&2; exit 1; }
    cd "$ROOT"
    if [[ $EUID -eq 0 ]]; then
      "$SING" build --force "$OUT/cfdpost.sif" containers/cfdpost.def
    else
      "$SING" build --force --fakeroot "$OUT/cfdpost.sif" containers/cfdpost.def
    fi
    ;;
  *)
    sed -n '2,10p' "$0"; exit 1
    ;;
esac
echo "done: $MODE ($MPI)"
