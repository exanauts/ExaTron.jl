module ExaTron

using Libdl
using LinearAlgebra
using CUDA
using CUDA.CUBLAS

export dtron, solveProblem, createProblem, setOption, getOption, ExaTronProblem

const BLAS_LIBRARY = :Tron
const EXATRON_LIBRARY = "libtron"

has_c_library() = !isnothing(Libdl.dlopen(EXATRON_LIBRARY; throw_error=false))
tron_zeros(S, n) = fill!(S(undef, Int64(n)), zero(eltype(S)))
tron_zeros(S, dims::Tuple) = fill!(S(undef, Int64(dims[1]), Int64(dims[1])), zero(eltype(S)))

include("utils.jl")
# BLAS routines
include("daxpy.jl")
include("dcopy.jl")
include("ddot.jl")
include("dmid.jl")
include("dnrm2.jl")
include("dscal.jl")
# Sparse routine
include("dssyax.jl")
# ICFS
include("ICFS/ICFS.jl")
using .ICFS
# Tron algorithm
include("dstrsol.jl")
include("dtrqsol.jl")
include("dbreakpt.jl")
include("dgpstep.jl")
include("dprsrch.jl")
include("dcauchy.jl")
include("dtrpcg.jl")
include("dspcg.jl")
include("dtron.jl")
include("driver.jl")

end # module
