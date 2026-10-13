#!/bin/bash
# Lovely-Haimes shock surface of one large solution across 4 nodes.
# Hybrid layout: 8 MPI ranks per node x 16 OpenMP threads (adapt to the node).
#SBATCH --job-name=cfdpost-shock
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00

module load gcc openmpi python          # the MPI cfdpost was built with
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores OMP_PROC_BIND=close

# --mpi=pmix (or pmi2) lets srun start MPI; cfdpost initialises MPI on import
srun --mpi=pmix --cpu-bind=cores python -m cfdpost shock flow.vtu -o shock_surface.vtp --preset default
