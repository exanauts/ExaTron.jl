function dgpnorm(n::Int, x::Array{T}, xl::Array{T},
                 xu::Array{T}, g::Array{T}) where T
    inf_norm = zero(T)
    for i=1:n
        if xl[i] != xu[i]
            if x[i] == xl[i]
                v = (min(g[i], zero(T)))^2
            elseif x[i] == xu[i]
                v = (max(g[i], zero(T)))^2
            else
                v = g[i]^2
            end

            v = sqrt(v)
            inf_norm = (inf_norm > v) ? inf_norm : v
        end
    end

    return inf_norm
end

@inline function dgpnorm(n::Int, x::CuDeviceArray{T,1}, xl::CuDeviceArray{T,1},
                         xu::CuDeviceArray{T,1}, g::CuDeviceArray{T,1}) where T
    tx = threadIdx().x

    v = zero(T)
    if tx <= n
        @inbounds begin
            if xl[tx] != xu[tx]
                if x[tx] == xl[tx]
                    v = min(g[tx], zero(T))
                elseif x[tx] == xu[tx]
                    v = max(g[tx], zero(T))
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
