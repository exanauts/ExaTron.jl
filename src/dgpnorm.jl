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

@inline function dgpnorm(n::Int, x::CuDeviceArray{Float64,1}, xl::CuDeviceArray{Float64,1},
                         xu::CuDeviceArray{Float64,1}, g::CuDeviceArray{Float64,1})
    tx = threadIdx().x

    v = 0.0
    if tx <= n
        @inbounds begin
            if xl[tx] != xu[tx]
                if x[tx] == xl[tx]
                    v = min(g[tx], 0.0)
                elseif x[tx] == xu[tx]
                    v = max(g[tx], 0.0)
                else
                    v = g[tx]
                end

                v = abs(v)
            end
        end
    end

    # shfl_down_sync() will automatically sync threads in a warp.

    offset = 16
    while offset > 0
        v = max(v, CUDA.shfl_down_sync(0xffffffff, v, offset))
        offset >>= 1
    end
    v = CUDA.shfl_sync(0xffffffff, v, 1)

    return v
end
