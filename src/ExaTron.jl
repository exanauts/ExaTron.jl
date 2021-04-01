module ExaTron

using Libdl
using LinearAlgebra
using CUDA
using CUDA.CUBLAS

using Printf
using DelimitedFiles

using Test
using Random

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

include("hs045_eval.jl")
include("launch.jl")
include("tron2_kernel.jl")

#=
include("opfdata.jl")
include("bus_kernel.jl")
include("generator_kernel.jl")
include("polar_kernel.jl")
include("tron_kernel.jl")
include("eval_kernel.jl")
include("auglag_kernel.jl")
include("acopf_admm_gpu.jl")
=#

end # module
