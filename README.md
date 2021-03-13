# ExaTron
This is a TRON solver implementation in Julia.
The intention is to make it work on GPUs as well.
Currently, we translated the Fortran implementation of [TRON](https://www.mcs.anl.gov/~more/tron)
into Julia.

# Performance ExaTron with ADMM on GPUs
Below is a table showing performance statistics of ExaTron used with ADMM on GPUs
over ACOPF problems without line limit.

| Data | Primal feasibility | Dual feasibility | Time (secs) |
| ---: | ---: | ---: | ---: |
|  case9241pegase | 2.404557e-03 | 8.329508e+00 | 145.96 |
| case13654pegase | 5.425782e-03 | 9.923688e+00 | 163.81 |

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



