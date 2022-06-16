@inline function ExaTron.dspcg(n::Int, delta::Float64, rtol::Float64, itermax::Int,
               x, xl,
               xu, A,
               g, s,
               B, L,
               indfree, gfree,
               w, iwa,
               wa1, wa2,
               wa3, wa4,
               wa5,
               tx)

    nfree = @localmem Int (1,)

    zero = 0.0
    one = 1.0

    # Compute A*(x[1] - x[0]) and store in w.

    dssyax(n, A, s, w, tx)

    # Compute the Cauchy point.

    daxpy(n,one,s,1,x,1,tx)
    dmid(n,x,xl,xu,tx)

    # Start the main iteration loop.
    # There are at most n iterations because at each iteration
    # at least one variable becomes active.

    info = 3
    iters = 0
    for nfaces=1:n

        # Determine the free variables at the current minimizer.
        # The indices of the free variables are stored in the first
        # n free positions of the array indfree.
        # The array iwa is used to detect free variables by setting
        # iwa[i] = nfree if the ith variable is free, otherwise iwa[i] = 0.

        # Use a single thread to avoid multiple branch divergences.
        # XXX: Would there be any gain in employing multiple threads?
        if tx == 1
            nfree[1] = 0
            @inbounds for j=1:n
                if xl[j] < x[j] && x[j] < xu[j]
                    nfree[1] = nfree[1] + 1
                    indfree[nfree[1]] = j
                    iwa[j] = nfree[1]
                else
                    iwa[j] = 0
                end
            end
        end
        @synchronize

        # Exit if there are no free constraints.

        if nfree[1] == 0
            info = 1
            return info, iters
        end

        # Obtain the submatrix of A for the free variables.
        # Recall that iwa allows the detection of free variables.
        reorder!(n, nfree[1], B, A, indfree, iwa, tx)

        # Compute the incomplete Cholesky factorization.
        alpha = zero
        dicfs(nfree[1], alpha, B, L, wa1, wa2, tx)

        # Compute the gradient grad q(x[k]) = g + A*(x[k] - x[0]),
        # of q at x[k] for the free variables.
        # Recall that w contains A*(x[k] - x[0]).
        # Compute the norm of the reduced gradient Z'*g.

        if tx <= nfree[1]
            @inbounds begin
                gfree[tx] = w[indfree[tx]] + g[indfree[tx]]
                wa1[tx] = g[indfree[tx]]
            end
        end
        @synchronize
        gfnorm = dnrm2(nfree[1],wa1,1,tx)

        # Save the trust region subproblem in the free variables
        # to generate a direction p[k]. Store p[k] in the array w.

        tol = rtol*gfnorm
        stol = zero

        infotr,itertr = dtrpcg(nfree[1],B,gfree,delta,L,
                               tol,stol,itermax,w,
                               wa1,wa2,wa3,wa4,wa5,tx)

        iters += itertr
        dtsol(nfree[1], L, w, tx)

        # Use a projected search to obtain the next iterate.
        # The projected search algorithm stores s[k] in w.

        if tx <= nfree[1]
            @inbounds begin
                wa1[tx] = x[indfree[tx]]
                wa2[tx] = xl[indfree[tx]]
                wa3[tx] = xu[indfree[tx]]
            end
        end
        @synchronize

        dprsrch(nfree[1],wa1,wa2,wa3,B,gfree,w,wa4,wa5,tx)

        # Update the minimizer and the step.
        # Note that s now contains x[k+1] - x[0].

        if tx <= nfree[1]
            @inbounds begin
                x[indfree[tx]] = wa1[tx]
                s[indfree[tx]] += w[tx]
            end
        end
        @synchronize

        # Compute A*(x[k+1] - x[0]) and store in w.

        dssyax(n, A, s, w, tx)

        # Compute the gradient grad q(x[k+1]) = g + A*(x[k+1] - x[0])
        # of q at x[k+1] for the free variables.

        if tx == 1
            @inbounds for j=1:nfree[1]
                gfree[j] = w[indfree[j]] + g[indfree[j]]
            end
        end
        @synchronize

        gfnormf = dnrm2(nfree[1], gfree, 1, tx)

        # Convergence and termination test.
        # We terminate if the preconditioned conjugate gradient
        # method encounters a direction of negative curvature, or
        # if the step is at the trust region bound.

        if gfnormf <= rtol*gfnorm
            info = 1
            return info, iters
        elseif infotr == 3 || infotr == 4
            info = 2
            return info, iters
        elseif iters > itermax
            info = 3
            return info, iters
        end
    end

    return info, iters
end
