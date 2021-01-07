"""
Subroutine dspcg

This subroutine generates a sequence of approximate minimizers
for the subproblem

    min { q(x) : xl <= x <= xu }.

The quadratic is defined by

    q(x[0]+s) = 0.5*s'*A*s + g'*s,

where x[0] is a base point provided by the user, A is a symmetric
matrix in compressed column storage, and g is a vector.

At each stage we have an approximate minimizer x[k], and generate
a direction p[k] by using a preconditioned conjugate gradient
method on the subproblem

    min { q(x[k]+p) : || L'*p || <= delta, s(fixed) = 0 },

where fixed is the set of variables fixed at x[k], delta is the
trust region bound, and L is an incomplete Cholesky factorization
of the submatrix

    B = A(free:free),

where free is the set of free variables at x[k]. Given p[k],
the next minimizer x[k+1] is generated by a projected search.

The starting point for this subroutine is x[1] = x[0] + s, where
x[0] is a base point and s is the Cauchy step.

The subroutine converges when the step s satisfies

    || (g + A*s)[free] || <= rtol*|| g[free] ||

In this case the final x is an approximate minimizer in the face
defined by the free variables.

The subroutine terminates when the trust region bound does
not allow further progress, that is, || L'*p[k] || = delta.
In this case the final x satisfies q(x) < q(x[k]).

MINPACK-2 Project. March 1999.
Argonne National Laboratory.
Chih-Jen Lin and Jorge J. More'.

March 2000

Clarified documentation of nv variable.
Eliminated the nnz = max(nnz,1) statement.
"""
function dspcg(n,x,xl,xu,A,g,delta,
               rtol,s,nv,itermax,
               B, L,
               indfree,gfree,w,wa,iwa)
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

        nfree = 0
        # TODO
        @inbounds for j=1:n
            if xl[j] < x[j] && x[j] < xu[j]
                nfree = nfree + 1
                indfree[nfree] = j
                iwa[j] = nfree
            else
                iwa[j] = 0
            end
        end

        # Exit if there are no free constraints.

        if nfree == 0
            info = 1
            return info, iters
        end

        # Obtain the submatrix of A for the free variables.
        # Recall that iwa allows the detection of free variables.
        nnz = reorder!(B, A, indfree, nfree, iwa)

        # Compute the incomplete Cholesky factorization.
        alpha = zero
        # TODO
        dicfs(nfree, nnz, B, L,
              nv, alpha,
              iwa, view(wa,1:n), view(wa,n+1:5*n))

        # Compute the gradient grad q(x[k]) = g + A*(x[k] - x[0]),
        # of q at x[k] for the free variables.
        # Recall that w contains A*(x[k] - x[0]).
        # Compute the norm of the reduced gradient Z'*g.

        # TODO
        @inbounds for j=1:nfree
            gfree[j] = w[indfree[j]] + g[indfree[j]]
            wa[j] = g[indfree[j]]
        end
        gfnorm = dnrm2(nfree,wa,1)

        # Save the trust region subproblem in the free variables
        # to generate a direction p[k]. Store p[k] in the array w.

        tol = rtol*gfnorm
        stol = zero

        infotr,itertr = dtrpcg(nfree,B,gfree,delta, L,
                               tol,stol,itermax,w,
                               view(wa,1:n),view(wa,n+1:2*n),view(wa,2*n+1:3*n),
                               view(wa,3*n+1:4*n),view(wa,4*n+1:5*n))

        iters = iters + itertr
        dtsol(nfree, L, w)

        # Use a projected search to obtain the next iterate.
        # The projected search algorithm stores s[k] in w.

        # TODO
        @inbounds for j=1:nfree
            wa[j] = x[indfree[j]]
            wa[n+j] = xl[indfree[j]]
            wa[2*n+j] = xu[indfree[j]]
        end

        dprsrch(nfree,view(wa,1:n),view(wa,n+1:2*n),view(wa,2*n+1:3*n),
                B,gfree,w,
                view(wa,3*n+1:4*n), view(wa,4*n+1:5*n))

        # Update the minimizer and the step.
        # Note that s now contains x[k+1] - x[0].

        # TODO
        @inbounds for j=1:nfree
            x[indfree[j]] = wa[j]
            s[indfree[j]] = s[indfree[j]] + w[j]
        end

        # Compute A*(x[k+1] - x[0]) and store in w.

        dssyax(n, A, s, w)

        # Compute the gradient grad q(x[k+1]) = g + A*(x[k+1] - x[0])
        # of q at x[k+1] for the free variables.

        # TODO
        @inbounds for j=1:nfree
            gfree[j] = w[indfree[j]] + g[indfree[j]]
        end
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
