module ExaTron

using Libdl
using LinearAlgebra
using CUDA
using CUDA.CUBLAS

using MPI

using Printf
using DelimitedFiles

#=
using JuMP
using Ipopt
using MathOptInterface
=#

export dtron, solveProblem, createProblem, setOption, getOption, ExaTronProblem

const BLAS_LIBRARY = :Tron
const EXATRON_LIBRARY = "libtron"

has_c_library() = !isnothing(Libdl.dlopen(EXATRON_LIBRARY; throw_error=false))
tron_zeros(S, n) = fill!(S(undef, Int64(n)), zero(eltype(S)))
tron_zeros(S, dims::Tuple) = fill!(S(undef, Int64(dims[1]), Int64(dims[1])), zero(eltype(S)))

include("daxpy.jl")
include("dcopy.jl")
include("ddot.jl")
include("dmid.jl")
include("dnrm2.jl")
include("dgpnorm.jl")
include("dscal.jl")
include("dsel2.jl")
include("dssyax.jl")
include("ihsort.jl")
include("insort.jl")
include("dnsol.jl")
include("dtsol.jl")
include("dtrqsol.jl")
include("dbreakpt.jl")
include("dgpstep.jl")
include("dicf.jl")
include("dicfs.jl")
include("dprsrch.jl")
include("dcauchy.jl")
include("dtrpcg.jl")
include("dspcg.jl")
include("dtron.jl")
include("driver.jl")

include("admm/parse_matpower.jl")
include("admm/opfdata.jl")
include("admm/environment.jl")
include("admm/check_violations.jl")
include("admm/model_io.jl")
#include("admm/environment_multiperiod.jl")
include("admm/bus_kernel.jl")
include("admm/generator_kernel.jl")
include("admm/eval_generator_kernel.jl")
include("admm/generator_kernel_multiperiod.jl")
include("admm/polar_kernel.jl")
include("admm/tron_kernel.jl")
include("admm/eval_kernel.jl")
include("admm/auglag_kernel.jl")
include("admm/eval_linelimit_kernel.jl")
include("admm/tron_linelimit_kernel.jl")
include("admm/auglag_linelimit_kernel.jl")
include("admm/acopf_admm_gpu.jl")
include("admm/acopf_admm_gpu_two_level.jl")
include("admm/acopf_admm_gpu_two_level_alternative.jl")
include("admm/acopf_admm_two_level_mpi_cpu.jl")
include("admm/acopf_admm_two_level_mpi.jl")
include("admm/acopf_admm_multiperiod_two_level_cpu.jl")
include("admm/acopf_admm_multiperiod_two_level_gpu.jl")
include("admm/acopf_admm_multiperiod.jl")
include("admm/acopf_admm_rolling_horizon_two_level_cpu.jl")
include("admm/acopf_admm_rolling_horizon_two_level_gpu.jl")
include("admm/acopf_admm_rolling_horizon.jl")
#include("admm/acopf_admm_gpu_multiperiod.jl")

#include("admm/test_multiperiod.jl")
#include("admm/test_parser.jl")
#include("admm/test_linelimit.jl")

function __init__()
    MPI.Init()
end

end # module
