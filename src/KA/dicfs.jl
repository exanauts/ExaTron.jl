@inline function ExaTron.dicfs(n::Int, alpha::Float64, A,
                       L,
                       wa1,
                       wa2,
                       tx)

    nbmax = 3
    alpham = 1.0e-3
    nbfactor = 512

    zero = 0.0
    one = 1.0
    two = 2.0

    # Compute the l2 norms of the columns of A.
    nrm2!(wa1, A, n, tx)

    # Compute the scaling matrix D.
    if tx <= n
        @inbounds wa2[tx] = (wa1[tx] > zero) ? one/sqrt(wa1[tx]) : one
    end
    @synchronize

    # Determine a lower bound for the step.

    if alpha <= zero
        alphas = alpham
    else
        alphas = alpha
    end

    # Compute the initial shift.

    alpha = zero
    for i in 1:n
        if A[i,i] == zero
            alpha = alphas
        else
            alpha = max(alpha, -A[i,i]*(wa2[i]^2))
        end
    end
    if alpha > 0
        alpha = max(alpha,alphas)
    end
    # if tx <= n  # No check on ty so that each warp has alpha.
    #     @inbounds alpha = (A[tx,tx] == zero) ? alphas : max(alpha, -A[tx,tx]*(wa2[tx]^2))
    # end

    # shfl_down_sync will automatically sync threads in a warp.

    # Find the maximum alpha in a warp and put it in the first thread.
    #offset = div(blockDim().x, 2)
    # offset = 16
    # while offset > 0
    #     alpha = max(alpha, CUDA.shfl_down_sync(0xffffffff, alpha, offset))
    #     offset >>= 1
    # end
    # # Broadcast it to the entire threads in a warp.
    # alpha = CUDA.shfl_sync(0xffffffff, alpha, 1)

    # if alpha > 0
    #     alpha = max(alpha,alphas)
    # end

    # Search for an acceptable shift. During the search we decrease
    # the lower bound alphas until we determine a lower bound that
    # is not acceptaable. We then increase the shift.
    # The lower bound is decreased by nbfactor at most nbmax times.

    nb = 1
    info = 0

    while true
        if tx <= n
            @inbounds for j=1:n
                L[j,tx] = A[j,tx] * wa2[j] * wa2[tx]
            end
            if alpha != zero
                @inbounds L[tx,tx] += alpha
            end
        end
        @synchronize

        # Attempt a Cholesky factorization.
        info = dicf(n, L, tx)

        # If the factorization exists, then test for termination.
        # Otherwise increment the shift.
        if info >= 0
            # If the shift is at the lower bound, reduce the shift.
            # Otherwise undo the scaling of L and exit.
            if alpha == alphas && nb < nbmax
                alphas /= nbfactor
                alpha = alphas
                nb = nb + 1
            else
                if tx <= n
                    @inbounds for j=1:n
                        if tx >= j
                            L[tx,j] /= wa2[tx]
                            L[j,tx] = L[tx,j]
                        end
                    end
                end
                @synchronize
                return
            end
        else
            alpha = max(two*alpha,alphas)
        end
    end
    @synchronize

    return
end
