module ExaTronKAKernels
    using ..ExaTron
    using ..KernelAbstractions
    const KA = KernelAbstractions
    include("architecture.jl")
    include("daxpy.jl")
    include("dcopy.jl")
    include("ddot.jl")
    include("dmid.jl")
    include("dnrm2.jl")
    include("dgpnorm.jl")
    include("dscal.jl")
    include("dssyax.jl")
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
    include("tron_qp_kernel.jl")
end

