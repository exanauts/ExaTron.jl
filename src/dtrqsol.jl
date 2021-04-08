"""
Subroutine dtrqsol

This subroutine computes the largest (non-negative) solution
of the quadratic trust region equation

  ||x + sigma*p|| = delta.

The code is only guaranteed to produce a non-negative solution
if ||x|| <= delta, and p != 0. If the trust region equation has
no solution, sigma = 0.

MINPACK-2 Project. March 1999.
Argonne National Laboratory.
Chih-Jen Lin and Jorge J. More'.
"""
function dtrqsol(n,x,p,delta)
    T = eltype(x)
    sigma = zero(T)

    ptx = ddot(n,p,1,x,1)
    ptp = ddot(n,p,1,p,1)
    xtx = ddot(n,x,1,x,1)
    dsq = delta^2

    # Guard against abnormal cases.
    rad = ptx^2 + ptp*(dsq - xtx)
    rad = sqrt(max(rad,zero(T)))

    if ptx > zero(T)
        sigma = (dsq - xtx)/(ptx + rad)
    elseif rad > zero(T)
        sigma = (rad - ptx)/ptp
    else
        sigma = zero(T)
    end

    return sigma
end

@inline function dtrqsol(n::Int,x::CuDeviceArray{T,1},
                         p::CuDeviceArray{T,1},delta::T) where T
    sigma = zero(T)

    ptx = ddot(n, p, 1, x, 1)
    ptp = ddot(n, p, 1, p, 1)
    xtx = ddot(n, x, 1, x, 1)
    dsq = delta^2

    # Guard against abnormal cases.
    rad = ptx^2 + ptp*(dsq - xtx)
    rad = sqrt(max(rad, zero(T)))

    if ptx > zero(T)
        sigma = (dsq - xtx)/(ptx + rad)
    elseif rad > zero(T)
        sigma = (rad - ptx)/ptp
    else
        sigma = zero(T)
    end
    CUDA.sync_threads()

    return sigma
end
