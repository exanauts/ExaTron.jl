# ExaTron.jl

This is a TRON solver implementation in Julia.
The intention is to make it work on GPUs as well.
Currently, we translated the Fortran implementation of [TRON](https://www.mcs.anl.gov/~more/tron)
into Julia.

## Installation

This package can be installed by cloning this repository:
```julia
] add https://github.com/exanauts/ExaTron.jl
```

## Performance ExaTron with ADMM on GPUs

With `@inbounds` attached to every array access and the use of instruction
parallelism instead of `for` loop, timings have reduced significantly.
The most recent results are as follows:

| Data | # active branches | Objective | Primal feasibility | Dual feasibility | Time (secs) | rho_pq | rho_va |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| case1354pegase  |  1991 | 7.400441e+04 | 1.186926e-05 | 9.799325e-03 |  24.78 |  10.0 |  1000.0 |
| case2869pegase  |  4582 | 1.338728e+05 | 1.831719e-04 | 3.570605e-02 |  42.86 |  10.0 |  1000.0 |
| case9241pegase  | 16049 | 3.139228e+05 | 2.526600e-03 | 8.328549e+00 |  98.88 |  50.0 |  5000.0 |
| case13659pegase | 20467 | 3.841941e+05 | 5.315441e-03 | 9.915973e+00 | 116.84 |  50.0 |  5000.0 |
| case19402_goc   | 34704 | 1.950577e+06 | 3.210911e-03 | 4.706196e+00 | 239.45 | 500.0 |  5000.0 |

For better accuracy, angle variables with constraints `\theta_i - \theta_j = \atan2(wI_{ij}, wR_{ij})`
were added.
This enables us to achieve a more accurate solution, since when there is a cycle in the network
the constraint forces that its sum of angles in the cycle is zero.
With new variables and constraints, experimental results are below.
We note that objective values have increased in most cases, which became closer to the values obtained
from Ipopt.

| Data | # active branches | Objective | Primal feasibility | Dual feasibility | Time (secs) | rho_pq | rho_va | # Iterations |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| case1354pegase  |  1,991 | 7.406087e+04 | 3.188928e-05 | 1.200796e-02 |  20.28 | 10.0 | 1000.0 | 5,000 |
| case2869pegase  |  4,582 | 1.339846e+05 | 2.123712e-04 | 2.228853e-01 |  35.74 | 10.0 | 1000.0 | 5,000 |
| case9241pegase  | 16,049 | 3.158906e+05 | 6.464865e-03 | 5.607324e+00 | 139.41 | 50.0 | 5000.0 | 6,000 |
| case13659pegase | 20,467 | 3.861735e+05 | 5.794895e-03 | 8.512909e+00 | 187.97 | 50.0 | 5000.0 | 7,000 |

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
