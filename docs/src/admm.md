# Distributed Optimization of ACOPF

This presents the use case of `ExaTron.jl` for solving large-scale alternating current optimal power flow (ACOPF) problem.
In this pacakge, we also provide the implementation of adaptive ADMM for distributed ACOPF introduced by [Mhanna et al. (2019)](https://doi.org/10.1109/TPWRS.2018.2886344). We have implemented the ADMM algorithm fully on GPUs without data transfer to the CPU, where `ExaTron.jl` is used to solve many small nonlinear nonconvex problems, each of which represents a branch subproblem of the ADMM.

## Numerical Experiment

All experiments were performed on a compute node of the Summit supercomputer at Oak Ridge
National Laboratory using `Julia@1.6.0` and `CUDA.jl@2.6.1`.
Note, however, that our implementation is not limited to a single node.
Each compute node of the Summit supercomputer has 2 sockets of POWER9 processors having 22 physical cores each, 512 GB of DRAM, and 6 NVIDIA Tesla V100 GPUs evenly distributed to each socket.

### ACOPF Problem Instances

The following table presents the data statistics of our test examples from MATPOWER and PGLIB benchmark instances.
We note that up to 34,000 nonlinear nonconvex problems are solved by our solver at each ADMM iteration.

| Data        | # Generators | # Branches | # Buses |
| ---------:  | -----------: | ---------: | ------: |
| 2868rte     | 600          | 3,808      | 2,868   |
| 6515rte     | 1,389        | 9,037      | 6,515   |
| 9241pegase  | 1,445        | 16,049     | 9,241   |
| 13659pegase | 4,092        | 20,467     | 13,659  |
| 19402goc    | 971          | 34,704     | 19,402  |

### Weak Scaling: Performance on a single GPU

The following figure depicts the average solution time of `ExaTron.jl` for different sizes of batches of branch subproblems.
The time on the y-axis is the average computation time in milliseconds taken by `ExaTron.jl` to solve each batch within an ADMM iteration.

![](single_gpu.pdf)

### Strong Scaling: Performance on multiple GPUs

The following figure shows the *speedup* of `ExaTron.jl` when we parallelize the computation across different GPUs (up to the 6 GPUs available on a node in the Summit supercomputer).
Branch problems are evenly dispatched among 6 MPI processes in the order of branch indices, and the
speedup is computed based on the timing of the root process.

![](multiple_gpus.pdf)

### Performance comparison: 6 GPUs vs. 40 CPUs

This experiment was run on a single Summit node with 6 GPUs and 40 CPUs.
For the CPU run, we use the MPI library to implement the parallel communication between the CPU processes.
In the following figure, the computation time of the CPU implementation shows a linear increase of with respect to the batch size.
However, the average computation time increases faster than that of the GPU implementation: the computation time of `ExaTron.jl` on 6 GPUs is up to 35 times faster than the CPU implementation using 40 cores.
Most of the speedup relates to the GPU's massive parallel computation capability.

![](cpu_vs_gpu.pdf)

## How to Reproduce the Numerical Results

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
