"""
Subroutine dtron

This subroutine implements a trust region Newton method for the
solution of large bound-constrained optimization problems

  min { f(x) : xl <= x <= xu }

where the Hessian matrix is sparse. The user must evaluate the
function, gradient, and the Hessian matrix.
"""
@inline function dtron(n, x, xl, xu, f, g, A, frtol, fatol,
                       fmin, cgtol, itermax, delta, task,
                       B, L, xc, s, indfree, gfree,
                       isave, dsave, wa, iwa,
                       wa1, wa2, wa3, wa4, wa5)
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

    if task == 0  # "START"

        # Initialize local variables.

        iter = 1
        iterscg = 0
        alphac = one
        work = 1  # "COMPUTE"

    else

        @inbounds begin
            # Restore local variables.

            work = isave[1]
            iter = isave[2]
            iterscg = isave[3]
            fc = dsave[1]
            alphac = dsave[2]
            prered = dsave[3]
        end
    end

    CUDA.sync_threads()

    # Search for a lower function value.

    search = true
    while search

        # Compute a step and evaluate the function at the trial point.

        if work == 1 # "COMPUTE"

            # Save the best function value, iterate, and gradient.

            fc = f
            dcopy(n,x,1,xc,1)

            # Compute the Cauchy step and store in s.

            alphac = dcauchy(n,x,xl,xu,A,g,delta,
                             alphac,s,wa)

            # Compute the projected Newton step.

            info,iters = dspcg(n, delta, cgtol, itermax,
                               x, xl, xu, A, g, s,
                               B, L,
                               indfree, gfree, wa, iwa,
                               wa1, wa2, wa3, wa4, wa5)

            # Compute the predicted reduction.

            dssyax(n, A, s, wa)
            prered = -(ddot(n,s,1,g,1) + p5*ddot(n,s,1,wa,1))
            iterscg = iterscg + iters

            # Set task to compute the function.

            task = 1 # 'F'
        end

        # Evaluate the step and determine if the step is successful.

        if work == 2 # "EVALUATE"

            # Compute the actual reduction.

            actred = fc - f

            # On the first iteration, adjust the initial step bound.

            snorm = dnrm2(n,s,1)
            if iter == 1
                delta = min(delta,snorm)
            end

            # Update the trust region bound.

            g0 = ddot(n,g,1,s,1)
            if f-fc-g0 <= zero
                alpha = sigma3
            else
                alpha = max(sigma1,-p5*(g0/(f-fc-g0)))
            end

            # Update the trust region bound according to the ratio
            # of actual to predicted reduction.

            if actred < eta0*prered
                delta = min(max(alpha,sigma1)*snorm,sigma2*delta)
            elseif actred < eta1*prered
                delta = max(sigma1*delta,min(alpha*snorm,sigma2*delta))
            elseif actred < eta2*prered
                delta = max(sigma1*delta,min(alpha*snorm,sigma3*delta))
            else
                delta = max(delta,min(alpha*snorm,sigma3*delta))
            end

            # Update the iterate.

            if actred > eta0*prered

                # Successful iterate.

                task = 2 # 'G' or 'H'
                iter = iter + 1

            else

                # Unsuccessful iterate.

                task = 1 # 'F'
                dcopy(n,xc,1,x,1)
                f = fc

            end

            # Test for convergence.

            if f < fmin
                task = 10 # "WARNING: F .LT. FMIN"
            end
            if abs(actred) <= fatol && prered <= fatol
                task = 4 # "CONVERGENCE: FATOL TEST SATISFIED"
            end
            if abs(actred) <= frtol*abs(f) && prered <= frtol*abs(f)
                task = 4 # "CONVERGENCE: FRTOL TEST SATISFIED"
            end
        end

        # Test for continuation of search

        if task == 1 && work == 2 # Char(task[1]) == 'F' && work == "EVALUATE"
            search = true
            work = 1 # "COMPUTE"
        else
            search = false
        end
    end

    if work == 3 # "NEWX"
        task = 3 # "NEWX"
    end

    # Decide on what work to perform on the next iteration.

    if task == 1 && work == 1 # Char(task[1]) == 'F' && work == "COMPUTE"
        work = 2 # "EVALUATE"
    elseif task == 1 && work == 2 # Char(task[1]) == 'F' && work == "EVALUATE"
        work = 1 # "COMPUTE"
    elseif task == 2 # unsafe_string(pointer(task),2) == "GH"
        work = 3 # "NEWX"
    elseif task == 3 # unsafe_string(pointer(task),4) == "NEWX"
        work = 1 # "COMPUTE"
    end

    @inbounds begin
        # Save local variables.

        isave[1] = work
        isave[2] = iter
        isave[3] = iterscg

        dsave[1] = fc
        dsave[2] = alphac
        dsave[3] = prered
    end

    CUDA.sync_threads()

    return delta, task
end
