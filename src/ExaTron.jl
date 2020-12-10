module ExaTron

using Libdl
using LinearAlgebra

export dtron, solveProblem, createProblem, addOption, ExaTronProblem

const BLAS_LIBRARY = :OpenBlas
const EXATRON_LIBRARY = "libtron"

has_c_library() = !isnothing(Libdl.dlopen(EXATRON_LIBRARY; throw_error=false))
tron_zeros(S, n) = fill!(S(undef, n), zero(eltype(S)))

include("daxpy.jl")
include("dcopy.jl")
include("ddot.jl")
include("dmid.jl")
include("dnrm2.jl")
include("dscal.jl")
include("dsel2.jl")
include("dssyax.jl")
include("ihsort.jl")
include("insort.jl")
include("dstrsol.jl")
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

end # module
