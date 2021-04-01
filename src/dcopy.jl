@inline function dcopy(n, dx, incx, dy, incy)
    tx = threadIdx().x
    ty = threadIdx().y

    # Ignore incx and incy for now.
    if tx <= n && ty == 1
        @inbounds dy[tx] = dx[tx]
    end
    CUDA.sync_threads()

    return
end