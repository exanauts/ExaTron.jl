@inline function ExaTron.dcopy(n::Int,dx::CuDeviceArray{Float64,1},incx::Int,
                       dy::CuDeviceArray{Float64,1},incy::Int)
    tx = threadIdx().x

    # Ignore incx and incy for now.
    if tx <= n
        @inbounds dy[tx] = dx[tx]
    end
    CUDA.sync_threads()

    return
end
