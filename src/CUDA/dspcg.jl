@inline function ExaTron.dspcg(n::Int, delta::Float64, rtol::Float64, itermax::Int,
               x::CuDeviceArray{Float64,1}, xl::CuDeviceArray{Float64,1},
               xu::CuDeviceArray{Float64,1}, A::CuDeviceArray{Float64,2},
               g::CuDeviceArray{Float64,1}, s::CuDeviceArray{Float64,1},
               B::CuDeviceArray{Float64,2}, L::CuDeviceArray{Float64,2},
               indfree::CuDeviceArray{Int,1}, gfree::CuDeviceArray{Float64,1},
               w::CuDeviceArray{Float64,1}, iwa::CuDeviceArray{Int,1},
               wa1::CuDeviceArray{Float64,1}, wa2::CuDeviceArray{Float64,1},
               wa3::CuDeviceArray{Float64,1}, wa4::CuDeviceArray{Float64,1},
               wa5::CuDeviceArray{Float64,1})

    tx = threadIdx().x

    zero = 0.0
    one = 1.0

    # Compute A*(x[1] - x[0]) and store in w.

    dssyax(n, A, s, w)

    # Compute the Cauchy point.

    daxpy(n,one,s,1,x,1)
    dmid(n,x,xl,xu)

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
        nfree = 0
        if tx == 1
            @inbounds for j=1:n
                if xl[j] < x[j] && x[j] < xu[j]
                    nfree = nfree + 1
                    indfree[nfree] = j
                    iwa[j] = nfree
                else
                    iwa[j] = 0
                end
            end
        end
        nfree = CUDA.shfl_sync(0xffffffff, nfree, 1)

        # Exit if there are no free constraints.

        if nfree == 0
            info = 1
            return info, iters
        end

        # Obtain the submatrix of A for the free variables.
        # Recall that iwa allows the detection of free variables.
        reorder!(n, nfree, B, A, indfree, iwa)

        # Compute the incomplete Cholesky factorization.
        alpha = zero
        dicfs(nfree, alpha, B, L, wa1, wa2)

        # Compute the gradient grad q(x[k]) = g + A*(x[k] - x[0]),
        # of q at x[k] for the free variables.
        # Recall that w contains A*(x[k] - x[0]).
        # Compute the norm of the reduced gradient Z'*g.

        if tx <= nfree
            @inbounds begin
                gfree[tx] = w[indfree[tx]] + g[indfree[tx]]
                wa1[tx] = g[indfree[tx]]
            end
        end
        CUDA.sync_threads()
        gfnorm = dnrm2(nfree,wa1,1)

        # Save the trust region subproblem in the free variables
        # to generate a direction p[k]. Store p[k] in the array w.

        tol = rtol*gfnorm
        stol = zero

        infotr,itertr = dtrpcg(nfree,B,gfree,delta,L,
                               tol,stol,itermax,w,
                               wa1,wa2,wa3,wa4,wa5)

        iters += itertr
        dtsol(nfree, L, w)

        # Use a projected search to obtain the next iterate.
        # The projected search algorithm stores s[k] in w.

        if tx <= nfree
            @inbounds begin
                wa1[tx] = x[indfree[tx]]
                wa2[tx] = xl[indfree[tx]]
                wa3[tx] = xu[indfree[tx]]
            end
        end
        CUDA.sync_threads()

        dprsrch(nfree,wa1,wa2,wa3,B,gfree,w,wa4,wa5)

        # Update the minimizer and the step.
        # Note that s now contains x[k+1] - x[0].

        if tx <= nfree
            @inbounds begin
                x[indfree[tx]] = wa1[tx]
                s[indfree[tx]] += w[tx]
            end
        end
        CUDA.sync_threads()

        # Compute A*(x[k+1] - x[0]) and store in w.

        dssyax(n, A, s, w)

        # Compute the gradient grad q(x[k+1]) = g + A*(x[k+1] - x[0])
        # of q at x[k+1] for the free variables.

        if tx == 1
            @inbounds for j=1:nfree
                gfree[j] = w[indfree[j]] + g[indfree[j]]
            end
        end
        CUDA.sync_threads()

        gfnormf = dnrm2(nfree, gfree, 1)

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
