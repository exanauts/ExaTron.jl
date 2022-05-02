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
