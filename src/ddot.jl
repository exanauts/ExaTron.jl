"""
Subroutine ddot

This subroutine forms the dot product of two vectors.
It uses unrolled loops for increments equal to one.
Jack Dongarra, LINPACK, 3/11/78.
"""
@inline function ddot(n, dx, incx, dy, incy)
    # Currently, all threads compute the same dot product,
    # hence, no sync_threads() is needed.
    # For very small n, we may want to gauge how much gains
    # we could get by run it in parallel.

    v = 0
    @inbounds for i=1:n
        v += dx[i]*dy[i]
    end
    CUDA.sync_threads()
    return v
end