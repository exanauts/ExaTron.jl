#!/bin/bash
#
# This script describes how to reproduce the results of Figure 8.
# This is just an example for iillustration purposes. Different platforms
# (such as Summit cluster) may require different setups.
#
# For each run of launch_mpi.jl, it will generate iteration logs
# and timing results. The relevant timing results for Figure 8 are printed
# at the end of its run and will be the following:
#
#   (Br+MPI)/iter = %.2f (millisecs)
#
# We use these numbers for the timings of 40 CPU cores and use the timings
# from Figure 6 for 6 GPUs.
#
# Prerequisite:
#  - CUDA library files should be accessible before executing this script,
#    e.g., module load cuda/10.2.89.
#  - CUDA aware MPI should be available.

export JULIA_CUDA_VERBOSE=1
export JULIA_MPI_BINARY="system"

DATA=("case2868rte" "case6515rte" "case9241pegase" "case13659pegase" "case19402_goc")
PQ=(10 20 50 50 500)
VA=(1000 2000 5000 5000 50000)
ITER=(6000 15000 35000 45000 30000)

for i in ${!DATA[@]}; do
	mpirun -np 40 julia --project ./src/launch_mpi.jl "./data/${DATA[$i]}" ${PQ[$i]} ${VA[$i]} ${ITER[$i]} false > output_cpu40_${DATA[$i]}.txt 2>&1
done
