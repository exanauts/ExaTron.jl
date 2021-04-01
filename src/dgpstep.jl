"""
Subroutine dgpstep

This subroutine computes the gradient projection step

  s = P[x + alpha*w] - x,

where P is the projection on the n-dimensional interval [xl,xu].
"""
@inline function dgpstep(n, x, xl, xu, alpha, w, s)
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