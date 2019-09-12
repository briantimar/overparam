#!/bin/bash
#SBATCH --job-name=ConvexGDDepthComparison
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --output=/home/btimar/code/overparam/ConvexGDDepthComparisons/slurm_logs
#SBATCH --time=01:00:00

pwd
date

echo "Training on all cores..."
julia -p 40 driver.jl
echo "Finished training"
date