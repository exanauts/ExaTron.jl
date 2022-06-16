@inline function ExaTron.dscal(n::Int,da::Float64,dx::CuDeviceArray{Float64,1},incx::Int)
    tx = threadIdx().x

    # Ignore incx for now.
    if tx <= n
        @inbounds dx[tx] = da*dx[tx]
    end
    CUDA.sync_threads()

    return
end