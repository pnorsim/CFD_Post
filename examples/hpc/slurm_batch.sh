#!/bin/bash
# Many design iterations: one rank per case (dynamic scheduling), 4 threads each.
#SBATCH --job-name=cfdpost-batch
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

module load gcc openmpi python
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# for very large cases give several ranks to each: add --group 8
srun --mpi=pmix python examples/batch_cases.py mesh.su2 DSN_*/ --outdir results
