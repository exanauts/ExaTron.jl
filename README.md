# ExaTron.jl

ExaTron.jl implements a trust-region Newton solver for batched nonlinear programming on GPUs.
Problems in the batch are solved in parallel by employing multiple thread blocks on GPUs.
Our basic algorithm to solve each problem on GPUs is an extension of the
algorithm by [Lin and More](https://doi.org/10.1137/S1052623498345075)
and its code [TRON](https://www.mcs.anl.gov/~more/tron).

## Installation

This package can be installed by cloning this repository:
```julia
] add https://github.com/exanauts/ExaTron.jl
```

## Usage: solving ACOPF using ADMM and ExaTron.jl on single GPU.

### On REPL
```julia
julia> using ExaTron
julia> env = ExaTron.admm_rect_gpu("./data/"*casename, T; iterlim=max_iter, rho_pq=rho_pq, rho_va=rho_va, use_gpu=use_gpu)
julia> ExaTron.admm_restart(env; iterlim=max_iter)
```
where
* `T`: Array type to use, `Array` for CPU or `CuArray` for GPU
* `caename`: the name of the test file of type `string`
* `rho_pq`: ADMM parameter for power flow of type `float`
* `rho_va`: ADMM parameter for voltage and angle of type `float`
* `max_iter`: maximum number of iterations of type `int`
* `use_gpu`: indicates whether to use gpu or not, of type `bool`

#### Example
```julia
julia> using ExaTron
julia> using CUDA
julia> env = ExaTron.admm_rect_gpu("./data/case9", CuArray; iterlim=2000, rho_pq=400.0, rho_va=40000.0, use_gpu=true)
```

## Usage: solving ACOPF using ADMM and ExaTron.jl on multiple GPUs.

In order to employ multiple GPUs, `MPI.jl` should be configured to work with `CuArray`.
We recommend to configure `Spack` for this.
```bash
$ git clone https://github.com/spack/spack.git
$ cd spack/
$ source share/spack/setup-env.sh
$ spack install openmpi +cuda
$ spack load openmpi
$ julia --project -e 'ENV["JULIA_MPI_BINARY"]="system"; using Pkg; Pkg.build("MPI"; verbose=true)'
```

## Data and parameter values
| Data | # Generators | # Branches | # Buses | rho_pq | rho_va | max_iter |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| case9 | 3 | 9 | 9 | 400.0 | 40000.0 | 2,000
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
