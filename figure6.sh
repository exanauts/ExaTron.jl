#!/bin/bash
#
# This script describes how to reproduce the results of Figure 6.
# This is just an example for iillustration purposes. Different platforms
# (such as Summit cluster) may require different setups.
#
# For each run of launch_mpi.jl, it will generate iteration logs
# and timing results. The relevant timing results for Figure 6 are printed
# at the end of its run and will be the following:
#
#   (Br+MPI)/iter = %.2f (millisecs)
#
# We divide the above timing results by the timing results obtained when
# we use a single GPU.
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
NGPU=(2 3 4 5 6)

for j in ${!NGPU[@]}; do
	for i in ${!DATA[@]}; do
		mpirun -np ${j} julia --project ./src/launch_mpi.jl "./data/${DATA[$i]}" ${PQ[$i]} ${VA[$i]} ${ITER[$i]} true > output_gpu${j}_${DATA[$i]}.txt 2>&1
        mv br_time_gpu.txt br_time_gpu${j}_${DATA[$i]}.txt
	done
done
