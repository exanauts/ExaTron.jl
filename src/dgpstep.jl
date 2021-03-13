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

function dgpstep(n::Int,x::CuDeviceArray{Float64,1},xl::CuDeviceArray{Float64,1},
                 xu::CuDeviceArray{Float64,1},alpha,w::CuDeviceArray{Float64,1},
                 s::CuDeviceArray{Float64,1})
    tx = threadIdx().x
    ty = threadIdx().y

    if tx <= n && ty == 1
        @inbounds begin
            # It might be better to process this using just a single thread,
            # rather than diverging between multiple threads.

            if x[tx] + alpha*w[tx] < xl[tx]
                s[tx] = xl[tx] - x[tx]
            elseif x[tx] + alpha*w[tx] > xu[tx]
                s[tx] = xu[tx] - x[tx]
            else
                s[tx] = alpha*w[tx]
            end
        end
    end
    CUDA.sync_threads()

    return
end