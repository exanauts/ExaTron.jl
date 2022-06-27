@inline function ExaTron.dprsrch(n::Int, x,
                         xl,
                         xu,
                         A,
                         g,
                         w,
                         wa1,
                         wa2,
                         tx)
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
    nbrpt,brptmin,brptmax = dbreakpt(n,x,xl,xu,w,tx)

    search = true
    while (search && alpha > brptmin)

        # Calculate P[x + alpha*w] - x and check the sufficient
        # decrease condition.

        nsteps = nsteps + 1
        dgpstep(n,x,xl,xu,alpha,w,wa1,tx)
        dssyax(n,A,wa1,wa2,tx)
        gts = ddot(n,g,1,wa1,1,tx)
        q = p5*ddot(n,wa1,1,wa2,1,tx) + gts
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

    dgpstep(n,x,xl,xu,alpha,w,wa1,tx)
    daxpy(n,alpha,w,1,x,1,tx)
    dmid(n,x,xl,xu,tx)
    dcopy(n,wa1,1,w,1,tx)

    return
end
