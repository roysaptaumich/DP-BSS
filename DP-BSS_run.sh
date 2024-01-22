#!/usr/bin/bash
#
# Author: Zehua Wang
# Updated: January 21th, 2024

# slurm options: --------------------------------------------------------------
#SBATCH --job-name=DP-BSS
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5GB
#SBATCH --time=1:00:00
#SBATCH --partition=standard
#SBATCH --array=1-10

# application: ----------------------------------------------------------------

# modules
module load gcc/10.3.0
module load armadillo/11.4.2
module load lapack/3.10.1

# compile program
g++ DP-BSS-ClusterParallel.cpp -o DP-BSS-eps1-${SLURM_ARRAY_TASK_ID} -O2 -I $ARMA_INC -DARMA_DONT_USE_WRAPPER -L $OPENBLAS_ROOT -lopenblas
# run program
./DP-BSS-eps1-${SLURM_ARRAY_TASK_ID}