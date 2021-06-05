#!/bin/bash
#
# This script describes how to reproduce the results of Figure 7.
# This is just an example for iillustration purposes. Different platforms
# (such as Summit cluster) may require different setups.
#
# We need br_time_13659pegase.txt file which is obtained when we run 
# with 6 GPUs over 13659pegase example. The file can be obtained by 
# running figure6.sh.

function usage() {
    echo "Usage: ./figure7.sh case"
    echo "  case: the case file containing branch computation time of each GPU"
}

if [[ $# != 1 ]]; then
    usage
    exit
fi

julia --project ./src/heatmap.jl $1
