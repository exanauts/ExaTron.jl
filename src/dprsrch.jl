"""
Subroutine dprsrch

This subroutine uses a projected search to compute a step
that satisfies a sufficient decrease condition for the quadratic

    q(s) = 0.5*s'*A*s + g'*s,

where A is a symmetric matrix in compressed column storage,
and g is a vector. Given the parameter alpha, the step is

    s[alpha] = P[x + alpha*w] - x,

where w is the search direction and P the projection onto the
n-dimensional interval [xl,xu]. The final step s = s[alpha]
satisfies the sufficient decrease condition

    q(s) <= mu_0*(g'*s),

where mu_0 is a constant in (0,1).

The search direction w must be a descent direction for the
quadratic q at x such that the quadratic is decreasing
in the ray x + alpha*w for 0 <= alpha <= 1.

MINPACK-2 Project. March 1999.
Argonne National Laboratory.
Chih-Jen Lin and Jorge J. More'.
"""
function dprsrch(n,x,xl,xu,A,g,w,wa1,wa2)
    one = 1.0
    p5 = 0.5

    # Constant that defines sufficient decrease.
    mu0 = 0.01

    # Interpolation factor.

    interpf = 0.5

    # Set the initial alpha = 1 because the quadratic function is
    # decreasing in the ray x + alpha*w for 0 <= alpha <= 1.

    alpha = one
    nsteps = 0

    # Find the smallest break-point on the ray x + alpha*w.
    nbrpt,brptmin,brptmax = dbreakpt(n,x,xl,xu,w)

    search = true
    while (search && alpha > brptmin)

        # Calculate P[x + alpha*w] - x and check the sufficient
        # decrease condition.

        nsteps = nsteps + 1
        dgpstep(n,x,xl,xu,alpha,w,wa1)
        dssyax(n, A,wa1,wa2)
        gts = ddot(n,g,1,wa1,1)
        q = p5*ddot(n,wa1,1,wa2,1) + gts
        if q <= mu0*gts
            search = false
        else

            # This is a crude interpolation procedure that
            # will be replaced in future versions of the code.

            alpha = interpf*alpha
        end
    end

    # Force at least one more constraint to be added to the active
    # set if alpha < brptmin and the full step is not successful.
    # There is sufficient decrease because the quadratic function
    # is decreasing in the ray x + alpha*w for 0 <= alpha <= 1.

    if (alpha < one && alpha < brptmin)
        alpha = brptmin
    end

    # Compute the final iterate and step.

    dgpstep(n,x,xl,xu,alpha,w,wa1)
    daxpy(n,alpha,w,1,x,1)
    dmid(n,x,xl,xu)
    dcopy(n,wa1,1,w,1)

    return
end
