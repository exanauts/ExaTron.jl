@inline function ExaTron.daxpy(n::Int,da::Float64,
                       dx::CuDeviceArray{Float64,1},incx::Int,
                       dy::CuDeviceArray{Float64,1},incy::Int)
    tx = threadIdx().x

    if tx <= n
        @inbounds dy[tx] = dy[tx] + da*dx[tx]
    end
    CUDA.sync_threads()

    return
end
