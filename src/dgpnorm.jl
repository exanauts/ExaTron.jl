@inline function dgpnorm(n, x, xl, xu, g)
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
