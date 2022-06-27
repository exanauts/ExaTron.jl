@inline function ExaTron.ddot(n::Int,dx::CuDeviceArray{Float64,1},incx::Int,
                      dy::CuDeviceArray{Float64,1},incy::Int)
    # Currently, all threads compute the same dot product,
    # hence, no sync_threads() is needed.
    # For very small n, we may want to gauge how much gains
    # we could get by run it in parallel.

    v = 0.0
    @inbounds for i=1:n
        v += dx[i]*dy[i]
    end
    return v
end
