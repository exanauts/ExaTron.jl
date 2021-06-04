#!/bin/bash
#
# This script describes how to reproduce the results of Table 5.
# This requires a br_time_gpu_case.txt file generated from figure6.sh.
# Since we compute the load imbalance based on the solution time,
# the values could be different from each run.

export JULIA_CUDA_VERBOSE=1
export JULIA_MPI_BINARY="system"

DATA=("2868rte" "6515rte" "9241pegase" "13659pegase" "19402goc")
for i in ${!DATA[@]}; do
    julia --project ./src/load_imbalance.jl "./br_time_gpu_${DATA[$i]}.txt"
done
