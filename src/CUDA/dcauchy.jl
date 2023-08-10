@inline function ExaTron.dcauchy(n::Int, x::CuDeviceArray{Float64,1},
                         xl::CuDeviceArray{Float64}, xu::CuDeviceArray{Float64,1},
                         A::CuDeviceArray{Float64,2}, g::CuDeviceArray{Float64,1},
                         delta::Float64, alpha::Float64, s::CuDeviceArray{Float64,1},
                         wa::CuDeviceArray{Float64,1})
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
