# ExaTron.jl

[![][build-release-img]][build-url]
[![DOI][doi-img]][doi-url]

ExaTron.jl implements a trust-region Newton solver for batched nonlinear programming on GPUs.
Problems in the batch are solved in parallel by employing multiple thread blocks on GPUs.
Our basic algorithm to solve each problem on GPUs is an extension of the
algorithm by [Lin and More](https://doi.org/10.1137/S1052623498345075)
and its code [TRON](https://www.mcs.anl.gov/~more/tron).

## Installation

This package can be installed by cloning this repository:
```julia
pkg> add ExaTron
```

## Usage: solving ACOPF using ADMM and ExaTron.jl on single GPU.
```julia
using ExaTron
using LazyArtifacts

# `datafile`: the name of the test file of type `String`
# here: MATPOWER case2868rte.m in ExaData project Artifact
datafile = joinpath(artifact"ExaData", "ExaData", "matpower", "case2868rte.m")
# `rho_pq`: ADMM parameter for power flow of type `Float64`
rho_pq = 10.0
# `rho_va`: ADMM parameter for voltage and angle of type `Float64`
rho_va = 1000.0
# `max_iter`: maximum number of iterations of type `Int`
max_iter = 5000
# `use_gpu`: indicates whether to use gpu or not, of type `Bool`
use_gpu = true
# Use polar formulation for branch problems
use_polar = true

env = ExaTron.admm_rect_gpu(datafile;
                      iterlim=max_iter, rho_pq=rho_pq, rho_va=rho_va, scale=1e-4, use_polar=use_polar, use_gpu=use_gpu)

# Restart and run 5000 iterations
ExaTron.admm_restart(env; iterlim=max_iter)
```

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

[build-release-img]: https://github.com/exanauts/ExaTron.jl/workflows/Run%20tests/badge.svg?branch=release
[build-url]: https://github.com/exanauts/ExaTron.jl/actions?query=workflow
[doi-img]: https://zenodo.org/badge/DOI/10.5281/zenodo.5829757.svg
[doi-url]: https://doi.org/10.5281/zenodo.5829757
