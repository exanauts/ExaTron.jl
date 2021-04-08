"""
Subroutine dmid

This subroutine computes the projection of x
on the n-dimensional interval [xl,xu].

MINPACK-2 Project. March 1999.
Argonne National Laboratory.
Chih-Jen Lin and Jorge J. More'.
"""
function dmid(n,x,xl,xu)
    @inbounds for i=1:n
        x[i] = max(xl[i],min(x[i],xu[i]))
    end
    return
end

@inline function dmid(n::Int, x::CuDeviceArray{T,1},
                      xl::CuDeviceArray{T,1}, xu::CuDeviceArray{T,1}) where T
    tx = threadIdx().x
    ty = threadIdx().y

    if tx <= n && ty == 1
        @inbounds x[tx] = max(xl[tx], min(x[tx], xu[tx]))
    end
    CUDA.sync_threads()

    return
end
