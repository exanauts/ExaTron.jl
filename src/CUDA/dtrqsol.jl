@inline function ExaTron.dtrqsol(n::Int,x::CuDeviceArray{Float64,1},
                         p::CuDeviceArray{Float64,1},delta::Float64)
    zero = 0.0
    sigma = zero

    ptx = ddot(n, p, 1, x, 1)
    ptp = ddot(n, p, 1, p, 1)
    xtx = ddot(n, x, 1, x, 1)
    dsq = delta^2

    # Guard against abnormal cases.
    rad = ptx^2 + ptp*(dsq - xtx)
    rad = sqrt(max(rad, zero))

    if ptx > zero
        sigma = (dsq - xtx)/(ptx + rad)
    elseif rad > zero
        sigma = (rad - ptx)/ptp
    else
        sigma = zero
    end
    CUDA.sync_threads()

    return sigma
end
