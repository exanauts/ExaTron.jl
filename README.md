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
