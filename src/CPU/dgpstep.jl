"""
Subroutine dgpstep

This subroutine computes the gradient projection step

  s = P[x + alpha*w] - x,

where P is the projection on the n-dimensional interval [xl,xu].
"""
function dgpstep(n,x,xl,xu,alpha,w,s)
    # TODO
    # This computation of the gradient projection step avoids
    # rouding errors for the components that are feasible.

    @inbounds for i=1:n
        if x[i] + alpha*w[i] < xl[i]
            s[i] = xl[i] - x[i]
        elseif x[i] + alpha*w[i] > xu[i]
            s[i] = xu[i] - x[i]
        else
            s[i] = alpha*w[i]
        end
    end
    return
end
