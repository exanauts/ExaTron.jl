@inline function ExaTron.dnrm2(n::Int,x::CuDeviceArray{Float64,1},incx::Int)
    tx = threadIdx().x

    v = 0.0
    if tx <= n  # No check on ty so that each warp has v.
        @inbounds v = x[tx]*x[tx]
    end

    # shfl_down_sync() will automatically sync threads in a warp.

    offset = 16
    while offset > 0
        v += CUDA.shfl_down_sync(0xffffffff, v, offset)
        offset >>= 1
    end
    v = sqrt(v)
    v = CUDA.shfl_sync(0xffffffff, v, 1)

    return v
end
