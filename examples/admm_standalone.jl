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

# Solve ACOPF
env = ExaTron.admm_rect_gpu(datafile;
                      iterlim=max_iter, rho_pq=rho_pq, rho_va=rho_va, scale=1e-4, use_polar=use_polar, use_gpu=use_gpu)

# Restart and run 5000 iterations
ExaTron.admm_restart!(env; iterlim=max_iter)