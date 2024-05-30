#!/bin/bash
#
# This script describes how to reproduce the results of Table 5.
# This requires a br_time_gpu_case.txt file generated from figure6.sh.
# Since we compute the load imbalance based on the solution time,
# the values could be different from each run.

export JULIA_CUDA_VERBOSE=1
export JULIA_MPI_BINARY="system"

function usage() {
    echo "Usage: ./table2.sh case"
    echo "  case: the case file containing branch computation time of each GPU"
}

if [[ $# != 1 ]]; then
    usage
    exit
fi

julia --project ./src/load_imbalance.jl $1
