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

We note that the following is for illustration purposes only.
If you want to run it on a HPC cluster, you may want to follow instructions specific to the HPC software.

### Using a single GPU

```bash
$ julia --project ./src/admm_standalone.jl ./data/casename pq_val va_val iterlim true
```
where `casename` is the filename of a power network, `pq_val` is an initial penalty value
for power values, `va_val` an initial penalty value for voltage values, `iterlim` the
maximum iteration limit, and `true|false` specifies whether to use GPU or CPU.
Power network files are provided in the `data` directory.

The following table shows what values need to be specified for parameters:

| casename | pq_val | va_val | iterlim |
| -------: | -----: | -----: | ------: |
| case2868rte | 10.0 | 1000.0 | 6,000 |
| case6515rte | 20.0 | 2000.0 | 15,000 |
| case9241pegase | 50.0 | 5000.0 | 35,000 |
| case13659pegase | 50.0 | 5000.0 | 45,000 |
| case19402_goc | 500.0 | 50000.0 | 30,000 |

For example, if you want to solve `case19402_goc` using a single GPU, you need to run
```bash
$ julia --project ./src/admm_standalone.jl ./data/case19402_goc 500 50000 30000 true
```

### Using multiple GPUs

If you want to use `N` GPUs, we launch `N` MPI processes and execute `launch_mpi.jl`.

```bash
$ mpirun -np N julia --project ./src/launch_mpi.jl ./data/casename pq_val va_val iterlim true
```

We assume that all of the MPI processes can see the `N` number of GPUs. Otherwise, it will generate an error.
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
For each figure or table, we provide a corresponding script to reproduce results.
Note that the following table shows correspondence between the casename and the size of batch.
| casename | batch size |
| -------: | ---------: |
| case2868rte | 3.8K |
| case6515rte | 9K |
| case9241pegase | 16K |
| case13659pegase | 20K |
| case19402_goc | 34K |

### Figure 5

```bash
$ ./figure5.sh
```

It will generate `output_gpu1_casename.txt` file for each `casename`. Near the end of the file, you will see
the timing results: `Branch/iter = %.2f (millisecs)` is the relevant result.
For example, in order to obtain timing results for `case19402_goc`, we read the following line around the end of 
the file
```bash
Branch/iter = 3.94 (millisecs)
```
Here `3.94` miiliseconds will be the input for the `34K` batch size in Figure 5.

### Figure 6

```bash
$ ./figure6.sh
```
It will generate `output_gpu${j}_casename.txt` file for each `casename` where `j` represents the number of GPUs
used. Near the end of the file, you will see the timing results: `[0] (Br+MPI)/iter = %.2f (millisecs)` is the relevant result,
where `[0]` represents the rank (the root in this case) of a process.
For example, in order to obtain timing results for `case19402_goc` with 6 GPUs, we read the following line around the end of the file
`output_gpu6_case19402_goc.txt`
```bash
[0] (Br+MPI)/iter = 0.79 (millisecs)
```
The speedup is `3.94/0.79 = 4.98` in this case. In this way, you can reproduce Figure 6.

### Table 5

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

### Figure 7

```bash
$ ./figure7.sh branch_time_file
```

The usage is the same as `table5.sh`. We use the same branch computation file.
To reproduce Figure 7, you need to use the file for `case13659pegase`:
```bash
$ ./figure7.sh br_time_gpu6_case13659pegase.txt
```
It will generate `br_time_gpu6_case13659pegase.pdf`. The file should be similar to Figure 7.

### Figure 8

```bash
$ ./figure8.sh
```

This script will run ExaTron.jl using 40 CPU cores. It will generate output files named `output_cpu40_casename.txt`.
Each file contains timing results for each case. For example, if you want to read the timing results for `case19402_goc`,
we read the following line around the end of the file.
```bash
[0] (Br+MPI)/iter = 30.03 (milliseconds)
```
Here `30.03` will be the input for `case19402_goc` for CPUs in Figure 8. For 6 GPUs, we use the results from `figure6.sh`.

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
