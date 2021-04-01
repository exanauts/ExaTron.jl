@inline function dscal(n, da, dx, incx)
    tx = threadIdx().x
    ty = threadIdx().y

    # Ignore incx for now.
    if tx <= n && ty == 1
        @inbounds dx[tx] = da*dx[tx]
    end
    CUDA.sync_threads()

    return
end