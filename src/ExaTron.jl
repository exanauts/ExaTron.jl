module ExaTron

using Libdl
using LinearAlgebra
using Requires
using Printf
using AMDGPU

export dtron, solveProblem, createProblem, setOption, getOption, ExaTronProblem, tron_qp_kernel
export daxpy, dcopy, ddot, dmid, dnrm2, dgpnorm, dscal, dssyax, dnsol, dtsol, dtrqsol, dbreakpt, dgpstep
export dicf, dicfs, dprsrch, dcauchy, dtrpcg, dspcg, dtron
export nrm2!, reorder!

const BLAS_LIBRARY = :Tron
const EXATRON_LIBRARY = "libtron"

has_c_library() = !isnothing(Libdl.dlopen(EXATRON_LIBRARY; throw_error=false))


include("TronMatrix.jl")
include("ihsort.jl")
include("insort.jl")
include("driver.jl")
include("dsel2.jl")

# Default CPU kernels
include("CPU/ExaTronCPUKernels.jl")

function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" include("CUDA/ExaTronCUDAKernels.jl")
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" using ExaTron.ExaTronCUDAKernels
    @require KernelAbstractions="63c18a36-062a-441e-b654-da1e3ab1ce7c" include("KA/ExaTronKAKernels.jl")
    @require KernelAbstractions="63c18a36-062a-441e-b654-da1e3ab1ce7c" using ExaTron.ExaTronKAKernels
end

end # module
