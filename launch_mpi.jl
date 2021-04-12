using ExaTron
using MPI
MPI.Init()

comm = MPI.COMM_WORLD

datafile = "./data/"*ARGS[1]
pq_val = parse(Float64, ARGS[2])
va_val = parse(Float64, ARGS[3])
max_iter = parse(Int, ARGS[4])
gpu = parse(Bool, ARGS[5])

# Warm-up
@time ExaTron.admm_rect_gpu_mpi(datafile; iterlim=1, rho_pq=pq_val, rho_va=va_val, use_gpu=gpu, use_polar=true, scale=1e-4)
# Computation
@time ExaTron.admm_rect_gpu_mpi(datafile; iterlim=max_iter, rho_pq=pq_val, rho_va=va_val, use_gpu=gpu, use_polar=true, scale=1e-4)

MPI.Finalize()

