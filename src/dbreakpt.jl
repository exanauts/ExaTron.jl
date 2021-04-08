"""
Subroutine dbreakpt

This subroutine computes the number of break-points, and
the minimal and maximal break-points of the projection of
x + alpha*w on the n-dimensional interval [xl,xu].
"""
function dbreakpt(n,x,xl,xu,w)
    T = eltype(x)
    nbrpt = 0
    brptmin = zero(T)
    brptmax = zero(T)

    @inbounds for i=1:n
        if (x[i] < xu[i] && w[i] > zero(T))
            nbrpt = nbrpt + 1
            brpt = (xu[i] - x[i]) / w[i]
            if nbrpt == 1
                brptmin = brpt
                brptmax = brpt
            else
                brptmin = min(brpt,brptmin)
                brptmax = max(brpt,brptmax)
            end
        elseif (x[i] > xl[i] && w[i] < zero(T))
            nbrpt = nbrpt + 1
            brpt = (xl[i] - x[i]) / w[i]
            if nbrpt == 1
                brptmin = brpt
                brptmax = brpt
            else
                brptmin = min(brpt,brptmin)
                brptmax = max(brpt,brptmax)
            end
        end
    end

    # Handle the exceptional case.

    if nbrpt == 0
        brptmin = zero(T)
        brptmax = zero(T)
    end

    return nbrpt,brptmin,brptmax
end

@inline function dbreakpt(n::Int, x::CuDeviceArray{T,1}, xl::CuDeviceArray{T,1},
                          xu::CuDeviceArray{T,1}, w::CuDeviceArray{T,1}) where T
    nbrpt = 0
    brptmin = zero(T)
    brptmax = zero(T)

    @inbounds for i=1:n
        if (x[i] < xu[i] && w[i] > zero(T))
            nbrpt = nbrpt + 1
            brpt = (xu[i] - x[i]) / w[i]
            if nbrpt == 1
                brptmin = brpt
                brptmax = brpt
            else
                brptmin = min(brpt,brptmin)
                brptmax = max(brpt,brptmax)
            end
        elseif (x[i] > xl[i] && w[i] < zero(T))
            nbrpt = nbrpt + 1
            brpt = (xl[i] - x[i]) / w[i]
            if nbrpt == 1
                brptmin = brpt
                brptmax = brpt
            else
                brptmin = min(brpt,brptmin)
                brptmax = max(brpt,brptmax)
            end
        end
    end

    # Handle the exceptional case.

    if nbrpt == 0
        brptmin = zero(T)
        brptmax = zero(T)
    end
    CUDA.sync_threads()

    return nbrpt,brptmin,brptmax
end
