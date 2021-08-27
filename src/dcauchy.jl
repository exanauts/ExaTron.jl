"""
Subroutine dcauchy

This subroutine computes a Cauchy step that satisfies a trust
region constraint and a sufficient decrease condition.

Ths Cauchy step is computed for the quadratic

    q(s) = 0.5*s'*A*s + g'*s

where A is a symmetric matrix in compressed row storage, and
g is a vector. Given a parameter alpha, the Cauchy step is

    s[alpha] = P[x - alpha*g] - x,

with P the projection onto the n-dimensional interval [xl,xu].
The Cauchy step satisfies the trust region constraint and the
sufficient decrease condition

    || s || <= delta,    q(s) <= mu_0*(g'*s),

where mu_0 is a constant in (0,1).

MINPACK-2 Project. March 1999.
Argonne National Laboratory.
Chih-Jen Lin and Jorge J. More'.
"""
function dcauchy(n,x,xl,xu,A,
                         g,delta,alpha,s,wa)
    p5 = 0.5
    one = 1.0

    # Constant that defines sufficient decrease.

    mu0 = 0.01

    # Interpolation and extrapolation factors.

    interpf = 0.1
    extrapf = 10.0

    # Find the minimal and maximal break-point on x - alpha*g.

    dcopy(n,g,1,wa,1)
    dscal(n,-one,wa,1)
    nbrpt,brptmin,brptmax = dbreakpt(n,x,xl,xu,wa)

    # Evaluate the initial alpha and decide if the algorithm
    # must interpolate or extrapolate.

    dgpstep(n,x,xl,xu,-alpha,g,s) # s = P(x - alpha*g) - x
    if dnrm2(n,s,1) > delta
        interp = true
    else
        dssyax(n, A, s, wa)
        gts = ddot(n,g,1,s,1)
        q = p5*ddot(n,s,1,wa,1) + gts
        interp = (q >= mu0*gts)
    end

    # Either interpolate or extrapolate to find a successful step.

    if interp

        # Reduce alpha until a successful step is found.

        search = true
        while search

            # This is a crude interpolation procedure that
            # will be replaced in future versions of the code.

            alpha = interpf*alpha
            dgpstep(n,x,xl,xu,-alpha,g,s)
            if dnrm2(n,s,1) <= delta
                dssyax(n, A, s,wa)
                gts = ddot(n,g,1,s,1)
                q = p5*ddot(n,s,1,wa,1) + gts
                search = (q > mu0*gts)
            end
        end

    else

        # Increase alpha until a successful step is found.

        search = true
        alphas = alpha
        while (search && alpha <= brptmax)

            # This is a crude extrapolation procedure that
            # will be replaced in future versions of the code.

            alpha = extrapf*alpha
            dgpstep(n,x,xl,xu,-alpha,g,s)
            if dnrm2(n,s,1) <= delta
                dssyax(n, A, s, wa)
                gts = ddot(n,g,1,s,1)
                q = p5*ddot(n,s,1,wa,1) + gts
                if q < mu0*gts
                    search = true
                    alphas = alpha
                end
            else
                search = false
            end
        end

        # Recover the last successful step.

        alpha = alphas
        dgpstep(n,x,xl,xu,-alpha,g,s)
    end

    return alpha
end

@inline function dcauchy(n::Int, x,
                         xl, xu,
                         A, g,
                         delta::Float64, alpha::Float64, s,
                         wa,
                         I, J)
    p5 = 0.5
    one = 1.0

    # Constant that defines sufficient decrease.

    mu0 = 0.01

    # Interpolation and extrapolation factors.

    interpf = 0.1
    extrapf = 10.0

    # Find the minimal and maximal break-point on x - alpha*g.

    dcopy(n,g,1,wa,1,I,J)
    dscal(n,-one,wa,1,I,J)
    nbrpt,brptmin,brptmax = dbreakpt(n,x,xl,xu,wa,I,J)

    # Evaluate the initial alpha and decide if the algorithm
    # must interpolate or extrapolate.

    dgpstep(n,x,xl,xu,-alpha,g,s,I,J) # s = P(x - alpha*g) - x
    if dnrm2(n,s,1,I,J) > delta
        interp = true
    else
        dssyax(n, A, s, wa,I,J)
        gts = ddot(n,g,1,s,1,I,J)
        q = p5*ddot(n,s,1,wa,1,I,J) + gts
        interp = (q >= mu0*gts)
    end

    # Either interpolate or extrapolate to find a successful step.

    if interp

        # Reduce alpha until a successful step is found.

        search = true
        while search

            # This is a crude interpolation procedure that
            # will be replaced in future versions of the code.

            alpha = interpf*alpha
            dgpstep(n,x,xl,xu,-alpha,g,s,I,J)
            if dnrm2(n,s,1,I,J) <= delta
                dssyax(n, A, s,wa,I,J)
                gts = ddot(n,g,1,s,1,I,J)
                q = p5*ddot(n,s,1,wa,1,I,J) + gts
                search = (q > mu0*gts)
            end
        end

    else

        # Increase alpha until a successful step is found.

        search = true
        alphas = alpha
        while (search && alpha <= brptmax)

            # This is a crude extrapolation procedure that
            # will be replaced in future versions of the code.

            alpha = extrapf*alpha
            dgpstep(n,x,xl,xu,-alpha,g,s,I,J)
            if dnrm2(n,s,1,I,J) <= delta
                dssyax(n, A, s, wa,I,J)
                gts = ddot(n,g,1,s,1,I,J)
                q = p5*ddot(n,s,1,wa,1,I,J) + gts
                if q < mu0*gts
                    search = true
                    alphas = alpha
                end
            else
                search = false
            end
        end

        # Recover the last successful step.

        alpha = alphas
        dgpstep(n,x,xl,xu,-alpha,g,s,I,J)
    end

    return alpha
end
