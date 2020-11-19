"""
Subroutine dtrpcg

Given a sparse symmetric matrix A in compressed column storage,
this subroutine uses a preconditioned conjugate gradient method
to find an approximate minimizer of the trust region subproblem

  min { q(s) : || L'*s || <= delta }.

where q is the quadratic

  q(s) = 0.5*s'*A*s + g'*s,

A is a symmetric matrix in compressed column storage, L is a
lower triangular matrix in compressed column storage, and g
is a vector.

This subroutine generates the conjugate gradient iterates for
the equivalent problem

  min { Q(w) : || w || <= delta },

where Q is the quadratic defined by

  Q(w) = q(s),        w = L'*s.

Termination occurs if the conjugate gradient iterates leave
the trust regoin, a negative curvature direction is generated,
or one of the following two convergence tests is satisfied.

Convergence in the original variables:

  || grad q(s) || <= tol

Convergence in the scaled variables:

  || grad Q(w) || <= stol

Note that if w = L'*s, then L*grad Q(w) = grad q(s).

MINPACK-2 Project. March 1999.
Argonne National Laboratory.
Chih-Jen Lin and Jorge J. More'.

August 1999

Corrected documentation for l, ldiag, lcol_ptr, and lrow_ind.

February 2001

We now set iters = 0 in the special case g = 0.
"""
function dtrpcg(n,A,g,delta, L,
                tol,stol,itermax,w,
                p,q,r,t,z)
    zero = 0.0
    one = 1.0

    # Initialize the iterate w and the residual r.

    for i=1:n
        w[i] = zero
    end

    # Initialize the residual t of grad q to -g.
    # Initialized the residual r of grad Q by solving L*r = -g.
    # Note that t = L*r.

    dcopy(n,g,1,t,1)
    dscal(n,-one,t,1)
    dcopy(n,t,1,r,1)
    dstrsol(n, L, r,'N')

    # Initialize the direction p.

    dcopy(n,r,1,p,1)

    # Initialize rho and the norms of r and t.

    rho = ddot(n,r,1,r,1)
    rnorm0 = sqrt(rho)

    # Exit if g = 0.

    iters = 0
    if rnorm0 == zero
        iters = 0
        info = 1
        return info, iters
    end

    for iters=1:itermax

        # Compute z by solving L'*z = p.

        dcopy(n,p,1,z,1)
        dstrsol(n, L, z,'T')

        # Compute q by solving L*q = A*z and save L*q for
        # use in updating the residual t.

        dssyax(n, A, z,q)
        dcopy(n,q,1,z,1)
        dstrsol(n, L, q,'N')

        # Compute alpha and determine sigma such that the trust region
        # constraint || w + sigma*p || = delta is satisfied.

        ptq = ddot(n,p,1,q,1)
        if ptq > zero
            alpha = rho/ptq
        else
            alpha = zero
        end
        sigma = dtrqsol(n,w,p,delta)

        # Exit if there is negative curvature or if the
        # iterates exit the trust region.

        if (ptq <= zero) || (alpha >= sigma)
            daxpy(n,sigma,p,1,w,1)
            if ptq <= zero
                info = 3
            else
                info = 4
            end

            return info, iters
        end

        # Update w and the residuals r and t.
        # Note that t = L*r.

        daxpy(n,alpha,p,1,w,1)
        daxpy(n,-alpha,q,1,r,1)
        daxpy(n,-alpha,z,1,t,1)

        # Exit if the residual convergence test is satisfied.

        rtr = ddot(n,r,1,r,1)
        rnorm = sqrt(rtr)
        tnorm = sqrt(ddot(n,t,1,t,1))

        if tnorm <= tol
            info = 1
            return info, iters
        end

        if rnorm <= stol
            info = 2
            return info, iters
        end

        # Compute p = r + beta*p and update rho.

        beta = rtr/rho
        dscal(n,beta,p,1)
        daxpy(n,one,r,1,p,1)
        rho = rtr
    end

    iters = itermax
    info = 5
    return info, iters
end
