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
