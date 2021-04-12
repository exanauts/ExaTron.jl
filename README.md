# ExaTron.jl

We implement a trust-region Newton solver for batched nonlinear programming on GPUs.
Our algorithm for a single optimization is based on the paper by
[Lin and More](https://doi.org/10.1137/S1052623498345075) and its code [TRON](https://www.mcs.anl.gov/~more/tron).

## Installation

This package can be installed by cloning this repository:
```julia
] add https://github.com/exanauts/ExaTron.jl
```

## Usage: solving ACOPF using ADMM and ExaTron.jl on single GPU.

### On command line
```bash
$ julia --project src/admm_standalone.jl casename rho_pq rho_va max_iter use_gpu
```

#### Example
```bash
$ julia --project src/admm_standalone.jl case2868rte 10 1000 5000 true
```

### On REPL
```julia
julia> using ExaTron
julia> ExaTron.admm_rect_gpu("./data/"*caesname; iterlim=max_iter, rho_pq=rho_pq, rho_va=rho_va, use_gpu=use_gpu)
```

## Usage: solving ACOPF using ADMM and ExaTron.jl on multiple GPUs.
```bash
$ mpiexec -n num julia --project launch_mpi.jl casename rho_pq rho_va max_iter use_gpu
```

## Data and parameter values
| Data | # Generators | # Branches | # Buses | rho_pq | rho_va | max_iter |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| case2868rte | 600 | 3,808 | 2,868 | 10.0 | 1000.0 | 5,000
| case6515rte | 1,389 | 9,037 | 6,515 | 20.0 | 2000.0 | 15,000
| case9241pegase | 1,445 | 16,049 | 9,241 | 50.0 | 5000.0 | 35,000
| case13659pegase | 4,092 | 20,467 | 13,659 | 50.0 | 5000.0 | 45,000
| case19402_goc | 971 | 34,704 | 19,402 | 500.0 | 50000.0 | 30,000

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
