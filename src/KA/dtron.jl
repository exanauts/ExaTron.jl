@inline function ExaTron.dtron(n::Int, x, xl,
               xu, f::Float64, g,
               A, frtol::Float64, fatol::Float64,
               fmin::Float64, cgtol::Float64, itermax::Int, delta::Float64, task::Int,
               B, L,
               xc, s,
               indfree, gfree,
               isave, dsave,
               wa, iwa,
               wa1, wa2,
               wa3, wa4,
               wa5,
               tx
               )
    zero = 0.0
    p5 = 0.5
    one = 1.0

    # Parameters for updating the iterates.

    eta0 = 1.0e-4
    eta1 = 0.25
    eta2 = 0.75

    # Parameters for updating the trust region size delta.

    sigma1 = 0.25
    sigma2 = 0.5
    sigma3 = 4.0

    work = 0

    # Initialization section.

        # Initialize local variables.

        iter = 1
        iterscg = 0
        alphac = one
        work = 1  # "COMPUTE"

    @synchronize

    # Search for a lower function value.

    # while search

        # Compute a step and evaluate the function at the trial point.


            # Save the best function value, iterate, and gradient.

            fc = f
            # dcopy(n,x,1,xc,1, tx)

            # Compute the Cauchy step and store in s.

            # AMDGPU.@rocprintf "bdcauchy x[1]: %s\n" x[1]
            alphac = dcauchy(n,x,xl,xu,A,g,delta,
                             alphac,s,wa, tx)
            # AMDGPU.@rocprintf "adcauchy x[1]: %s\n" x[1]
            @synchronize
            #C AMDGPU.@rocprintf "adcauchy x[1]: %s < %s < %s wa[1]: %s\n" xl[1] x[1] xu[1] wa1[1]
            AMDGPU.@rocprintf "adcauchy x[1]: %s < %s < %s wa[1]: %s\n" xl[1] x[1] xu[1] wa1[1]


            # Compute the projected Newton step.

            # AMDGPU.@rocprintf "bdspcg x[1]: %s\n" x[1]
            info,iters = dspcg(n, delta, cgtol, itermax,
                               x, xl, xu, A, g, s,
                               B, L,
                               indfree, gfree, wa, iwa,
                               wa1, wa2, wa3, wa4, wa5, tx)
            @synchronize
            #C AMDGPU.@rocprintf :lane "adspcg wa1[1] x[1]: %s %s %s \n" wa1[1] x[1] info
            AMDGPU.@rocprintf :lane "adspcg wa1[1] x[1]: %s %s %s \n" wa1[1] x[1] info

            # # Compute the predicted reduction.

            dssyax(n, A, s, wa, tx)
            # prered = -(ddot(n,s,1,g,1,tx) + p5*ddot(n,s,1,wa,1,tx))
            # iterscg = iterscg + iters

            # Set task to compute the function.

            task = 1 # 'F'

        # Evaluate the step and determine if the step is successful.


    @synchronize

    return delta, task
end
