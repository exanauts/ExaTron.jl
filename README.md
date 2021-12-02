# ExaTron.jl

[![Run tests](https://github.com/exanauts/ExaTron.jl/actions/workflows/action.yml/badge.svg)](https://github.com/exanauts/ExaTron.jl/actions/workflows/action.yml)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://exanauts.github.io/ExaTron.jl/dev)

ExaTron is the Julia implementaion of novel GPU-accelerated algorithm for bound-constrained nonlinear nonconvex optimization problems of the form

<!-- $$
\min_x \; f(x) \; \text{subject to} \; l \leq x \leq u,
$$ --> 

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cmin_x%20%5C%3B%20f(x)%20%5C%3B%20%5Ctext%7Bsubject%20to%7D%20%5C%3B%20l%20%5Cleq%20x%20%5Cleq%20u%2C"></div>

where <!-- $x \in \mathbf{R}^d$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x%20%5Cin%20%5Cmathbf%7BR%7D%5Ed"> is the optimization variable and <!-- $l,u \in \mathbf{R}^d \cup \{-\infty,\infty\}^d$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=l%2Cu%20%5Cin%20%5Cmathbf%7BR%7D%5Ed%20%5Ccup%20%5C%7B-%5Cinfty%2C%5Cinfty%5C%7D%5Ed"> are respectively lower and upper bounds (allowing negative and positive infinite values).
Bound constraints hold componentwise, and the objective function <!-- $f: \mathbf{R}^d \rightarrow \mathbf{R}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=f%3A%20%5Cmathbf%7BR%7D%5Ed%20%5Crightarrow%20%5Cmathbf%7BR%7D"> is a generic nonlinear nonconvex function.
Bound-constrained problems play an important role as a building block to solve problems with more general constraints such as <!-- $h(x)=0$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=h(x)%3D0">, where <!-- $h$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=h"> is a linear or a nonlinear function.
The algorithm is a variant of [TRON (Lin and Moré, 1999)](https://www.mcs.anl.gov/~more/tron) with the complete Cholesky factorization for preconditioning, which has been carefully designed for solving extremely many small nonlinear problems as GPU batching.

## Installation

This package can be installed by cloning this repository:
```julia
] add https://github.com/exanauts/ExaTron.jl
```

## Use Case: Distributed Optimization of ACOPF

This presents the use case of `ExaTron.jl` for solving large-scale alternating current optimal power flow (ACOPF) problem.
In this pacakge, we also provide the implementation of adaptive ADMM for distributed ACOPF introduced by [Mhanna et al. (2019)](https://doi.org/10.1109/TPWRS.2018.2886344). We have implemented the ADMM algorithm fully on GPUs without data transfer to the CPU, where `ExaTron.jl` is used to solve many small nonlinear nonconvex problems, each of which represents a branch subproblem of the ADMM. See details in the documentation.

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
