

# Utils
struct InnerIterationInfo{T}
    bus_time::T
    gen_time::T
    branch_time::T
    primal_resid::T
    dual_resid::T
    z_norm::T
    mismatch::T
    eps_primal::T
end

