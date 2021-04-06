using ExaTron
using MPI
MPI.Init()

comm = MPI.COMM_WORLD

datafile = "/home/fpacaud/exa/proxALM/data/case9241pegase"
# datafile = "/home/fpacaud/exa/proxALM/data/case9"

gpu = false
# Warm-up
@time ExaTron.admm_rect_gpu_mpi(datafile; iterlim=1, rho_pq=50, rho_va=5000, use_gpu=gpu, gpu_no=1, use_polar=true)
# Computation
@time ExaTron.admm_rect_gpu_mpi(datafile; iterlim=10, rho_pq=50, rho_va=5000, use_gpu=gpu, gpu_no=1, use_polar=true)

MPI.Finalize()

