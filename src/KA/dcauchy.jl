@inline function ExaTron.dcauchy(n::Int, x,
                         xl, xu,
                         A, g,
                         delta::Float64, alpha::Float64, s,
                         wa,
                         tx)
    p5 = 0.5
    one = 1.0

    # Constant that defines sufficient decrease.

    mu0 = 0.01

    # Interpolation and extrapolation factors.

    interpf = 0.1
    extrapf = 10.0

    # Find the minimal and maximal break-point on x - alpha*g.

    # dcopy(n,g,1,wa,1,tx)
    # dscal(n,-one,wa,1,tx)
    # nbrpt,brptmin,brptmax = dbreakpt(n,x,xl,xu,wa,tx)

    # Evaluate the initial alpha and decide if the algorithm
    # must interpolate or extrapolate.

    dgpstep(n,x,xl,xu,-alpha,g,s,tx) # s = P(x - alpha*g) - x
    lnrm = dnrm2(n,s,1,tx)
    interp = true
    # if lnrm > delta
    # else
    #     dssyax(n, A, s, wa,tx)
    #     gts = ddot(n,g,1,s,1,tx)
    #     q = p5*ddot(n,s,1,wa,1,tx) + gts
    #     interp = (q >= mu0*gts)
    # end

    @synchronize
    # Either interpolate or extrapolate to find a successful step.
    # if tx == 1
        # AMDGPU.@rocprintf "interp: %s lnrm: %s delta: %s dim: %s\n" interp lnrm delta AMDGPU.workgroupDim()
        # AMDGPU.@rocprintf "x[1]: %s x[2]: %s\n" x[1] x[2]
        # AMDGPU.@rocprintf "w[1]: %s w[2]: %s\n" g[1] g[2]
        # AMDGPU.@rocprintf "xl[1]: %s xl[2]: %s\n" xl[1] xl[2]
        # AMDGPU.@rocprintf "xu[1]: %s xu[2]: %s\n" xu[1] xu[2]
        # AMDGPU.@rocprintf "alpha: %s\n" alpha
        # AMDGPU.@rocprintf "dcauchy x[1]: %s interp: %s lnrm: %s delta: %s n: %s s[1]: %s s[2]: %s\n" x[1] interp lnrm delta n s[1] s[2]
        #C AMDGPU.@rocprintf "dcauchy x[1]: %s delta: %s n: %s s[1]: %s s[2]: %s\n" x[1] delta n s[1] s[2]
        AMDGPU.@rocprintf "dcauchy x[1]: %s delta: %s n: %s s[1]: %s s[2]: %s\n" x[1] delta n s[1] s[2]
    # end
    # if interp

        # Reduce alpha until a successful step is found.

        # search = true
        # while search

        #     # This is a crude interpolation procedure that
        #     # will be replaced in future versions of the code.

        #     alpha = interpf*alpha
        #     dgpstep(n,x,xl,xu,-alpha,g,s,tx)
        #     if dnrm2(n,s,1,tx) <= delta
        #         dssyax(n, A, s,wa,tx)
        #         gts = ddot(n,g,1,s,1,tx)
        #         q = p5*ddot(n,s,1,wa,1,tx) + gts
        #         search = (q > mu0*gts)
        #     end
        # end

    # else

    #     # Increase alpha until a successful step is found.

    #     search = true
    #     alphas = alpha
    #     while (search && alpha <= brptmax)

    #         # This is a crude extrapolation procedure that
    #         # will be replaced in future versions of the code.

    #         alpha = extrapf*alpha
    #         dgpstep(n,x,xl,xu,-alpha,g,s,tx)
    #         if dnrm2(n,s,1,tx) <= delta
    #             dssyax(n, A, s, wa,tx)
    #             gts = ddot(n,g,1,s,1,tx)
    #             q = p5*ddot(n,s,1,wa,1,tx) + gts
    #             if q < mu0*gts
    #                 search = true
    #                 alphas = alpha
    #             end
    #         else
    #             search = false
    #         end
    #     end

    #     # Recover the last successful step.

    #     alpha = alphas
    #     dgpstep(n,x,xl,xu,-alpha,g,s,tx)
    # end

    return alpha
end
