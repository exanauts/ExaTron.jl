# ExaTron.jl

ExaTron.jl implements a trust-region Newton algorithm for bound constrained batch nonlinear
programming on GPUs.
Its algorithm is based on [Lin and More](https://epubs.siam.org/doi/10.1137/S1052623498345075)
and [TRON](https://www.mcs.anl.gov/~more/tron).

## Installation

This package can be installed by cloning this repository:
```julia
] add https://github.com/exanauts/ExaTron.jl
```

## How to run

We describe how to run ExaTron on the [Summit](https://docs.olcf.ornl.gov/systems/summit_user_guide.html) cluster
at Oak Ridge National Laboratory.


### Using a single GPU

```bash
#!/bin/bash
#BSUB -P your_project_code
#BSUB -W 0:30
#BSUB -nnodes 1
#BSUB -J ExaTron
#BSUB -o ExaTron.%J

cd $PROJWORK/your_ExaTron_directory
date
module load gcc/7.4.0
module load cuda/10.2.89
export JULIA_CUDA_VERBOSE=1
export JULIA_DEPOT_PATH=your_julia_depot_path
export JULIA_MPI_BINARY=system
jsrun -n 1 -r 1 -a 1 -c 1 -g 1 julia --project ./src/admm_standalone.jl ./data/casename pq_val va_val iterlim true
```
where `casename` is the filename of a power network, `pq_val` is an initial penalty value
for power values, `va_val` an initial penalty value for voltage values, `iterlim` the
maximum iteration limit, and `true|false` specifies whether to use GPU or CPU.
Power network files are provided in the `data` directory.
In the line starting with `jsrun`, you may want to replace `julia` with its absolute path.

The following table shows what values need to be specified for parameters:

| casename | pq_val | va_val | iterlim |
| -------: | -----: | -----: | ------: |
| case2868rte | 10.0 | 1000.0 | 6,000 |
| case6515rte | 20.0 | 2000.0 | 15,000 |
| case9241pegase | 50.0 | 5000.0 | 35,000 |
| case13659pegase | 50.0 | 5000.0 | 45,000 |
| case19402_goc | 500.0 | 50000.0 | 30,000 |

For example, if you want to solve `case19402_goc` using a single GPU, replace the line starting
with `jsrun` with the following
```bash
jsrun -n 1 -r 1 -a 1 -c 1 -g 1 julia --project ./src/admm_standalone.jl ./data/case19402_goc 500 50000 30000 true
```

### Using multiple GPUs

If you want to use `N` GPUs, replace the line starting with `jsrun` with the following:

```bash
jsrun --smpiargs="-gpu" -n 1 -r 1 -a N -c N -g N -d packed julia --project ./src/launch_mpi.jl ./data/casename pq_val va_val iterlim true
```

The parameter values are the same as the single GPU case, except that we use the following actual
iteration limit for each case. If you see the logs, the total number of iterations is the same as single GPU case.
| casename | iterlim |
| -------: | ------: |
| case2868rte | 5648 |
| case6515rte | 13651 |
| case9241pegase | 30927 |
| case13659pegase | 41126 |
| case19402_goc | 28358 |

## Reproducing experiments

We describe how to reproduce experiments in Section 6 of our manuscript.
Note that the following table shows correspondence between the casename and the size of a batch.
| casename | batch size |
| -------: | ---------: |
| case2868rte | 3.8K |
| case6515rte | 9K |
| case9241pegase | 16K |
| case13659pegase | 20K |
| case19402_goc | 34K |

### Figure 10

To reproduce Figure 5, submit a job with each case file and its parameter values.
For each case with name `casename`, it will generate `output_gpu1_casename.txt`.
Near the end of the file, you will see the timing results: `Branch/iter = %.2f (millisecs)` is the relevant result.
For example, in order to obtain timing results for `case19402_goc`, we read the following line around the end of
the file
```bash
Branch/iter = 3.94 (millisecs)
```
Here `3.94` miiliseconds will be the input for the `34K` batch size in Figure 5.

### Figure 11

To reproduce Figure 6, submit a job with each case file, its parameter values, and different GPU number `N`.
It will generate `output_gpu${N}_casename.txt` file for each `casename` where `N` represents the number of GPUs
used.
Near the end of the file, you will see the timing results: `[0] (Br+MPI)/iter = %.2f (millisecs)` is the relevant result,
where `[0]` represents the rank (the root in this case) of a process.
For example, in order to obtain timing results for `case19402_goc` with 6 GPUs, we read the following line around the end of the file
`output_gpu6_case19402_goc.txt`
```bash
[0] (Br+MPI)/iter = 0.79 (millisecs)
```
The speedup is `3.94/0.79 = 4.98` in this case. In this way, you can reproduce Figure 6.

### Table 2

```bash
$ ./table5.sh branch_time_file
```
where `branch_time_file` corresponds to the file containing the computation time of branch of each GPU.
It is generated from `figure6.sh`. For example, `figure6.sh` will generate the following files:
```bash
br_time_gpu6_case2868rte.txt
br_time_gpu6_case6515rte.txt
br_time_gpu6_case9241pegase.txt
br_time_gpu6_case13659pegase.txt
br_time_gpu6_case19402_goc.txt
```
The following command will give you the load imbalance statistics for `case13659pegase`:
```bash
$ ./table5.sh br_time_gpu6_case13659pegase.txt
```
Similarly, you can reproduce load imbalance statistics for other case files.

### Figure 12

```bash
$ ./figure7.sh branch_time_file
```

The usage is the same as `table5.sh`. We use the same branch computation file.
To reproduce Figure 7, you need to use the file for `case13659pegase`:
```bash
$ ./figure7.sh br_time_gpu6_case13659pegase.txt
```
It will generate `br_time_gpu6_case13659pegase.pdf`. The file should look similar to Figure 7.

### Figure 13

To reproduce Figure 8, we need to execute ExaTron with 40 CPU cores.
For this, we replace the line starting with `jsrun` with the following:
```bash
jsrun -n 1 -r 1 -a 40 -c 40 -g 0 -d packed julia --project ./src/launch_mpi.jl ./data/casename pq_val va_val iterlim false
```

We use the following `iterlim` for CPU runs. We note that the value of `iterlim` is different than when we use `GPUs`.
This is because float-point computations are performed on different hardware, leading to slightly different results.

| casename | iterlim |
| -------: | ------: |
| case2868rte | 5718 |
| case6515rte | 13640 |
| case9241pegase | 30932 |
| case13659pegase | 41140 |
| case19402_goc | 28358 |

It will generate output files named `output_cpu40_casename.txt`.
Each file contains timing results for each case. For example, if you want to read the timing results for `case19402_goc`,
we read the following line around the end of the file.
```bash
[0] (Br+MPI)/iter = 30.03 (milliseconds)
```
Here `30.03` will be the input for `case19402_goc` for CPUs in Figure 8. For 6 GPUs, we use the results from `figure6.sh`.

### Running ExaTron directly on a non-cluster machine

If you want to run ExaTron on a non-cluster, copy `julia --project ...` part in the line containing `jsrun`.
For multiple GPUs, run with `mpirun -np N julia --project ..`
Note that all of the MPI processes should be able to see the `N` number of GPUs. Otherwise, it will generate an error.

### Generating PTX code for a kernel

By running the following, you could generate PTX code for a kernel:
```bash
@device_code_ptx CUDA.@sync @cuda threads=32 blocks=10240 kernel_func(a,b)
```
where the numbers for `threads` and `blocks` and the arguments `a` and `b` depend on `kernel_func`.
If needed, you may want to specify its shared memory size.

## Citing this package

```
@misc{ExaTron.jl.0.0.0,
  author       = {Kim, Youngdae and Pacaud, Fran\ccois and Kim, Kibaek},
  title        = {{ExaTron.jl: GPU-capable TRON solver in Julia}},
  month        = Mar,
  year         = 2021,
  version      = {0.0.0},
  url          = {https://github.com/exanauts/ExaTron.jl}
}
```

## Acknowledgements

This material is based upon work supported by the U.S. Department of Energy, Office of Science, under contract number DE-AC02-06CH11357.
