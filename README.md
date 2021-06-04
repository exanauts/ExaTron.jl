# ExaTron.jl

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
