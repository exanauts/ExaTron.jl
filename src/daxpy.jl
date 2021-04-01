"""
Subroutine daxpy

This subroutine computes constant times a vector plus a vector.
It uses unrolled loops for increments equal to one.
Jack Dongarra, LINPACK, 3/11/78.
"""
@inline function daxpy(n, da, dx, incx, dy, incy)
    tx = threadIdx().x
    ty = threadIdx().y

    if tx <= n && ty == 1
        @inbounds dy[tx] = dy[tx] + da*dx[tx]
    end
    CUDA.sync_threads()

    return
end