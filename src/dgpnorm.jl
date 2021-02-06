function dgpnorm(n::Int, x::Array{Float64}, xl::Array{Float64},
                 xu::Array{Float64}, g::Array{Float64})
    inf_norm = 0.0
    for i=1:n
        if xl[i] != xu[i]
            if x[i] == xl[i]
                v = (min(g[i], 0.0))^2
            elseif x[i] == xu[i]
                v = (max(g[i], 0.0))^2
            else
                v = g[i]^2
            end

            v = sqrt(v)
            inf_norm = (inf_norm > v) ? inf_norm : v
        end
    end

    return inf_norm
end

function dgpnorm(n::Int, x::CuDeviceArray{Float64}, xl::CuDeviceArray{Float64},
                 xu::CuDeviceArray{Float64}, g::CuDeviceArray{Float64})
    tx = threadIdx().x
    ty = threadIdx().y

    # Q: Would it be better to use a single thread?
    # A: It is tricky to share values between threads.
    # Q: Would it be better to go througbh in ty direction?
    # A: No, the runtime was similar.
    v = 0.0
    if tx <= n
        if xl[tx] != xu[tx]
            if x[tx] == xl[tx]
                v = (min(g[tx], 0.0))^2
            elseif x[tx] == xu[tx]
                v = (max(g[tx], 0.0))^2
            else
                v = g[tx]^2
            end

            v = sqrt(v)
        end
    end

    offset = div(blockDim().x, 2)
    while offset > 0
        v = max(v, CUDA.shfl_down_sync(0xffffffff, v, offset))
        offset = div(offset, 2)
    end

    v = CUDA.shfl_sync(0xffffffff, v, 1)
    return v
end
