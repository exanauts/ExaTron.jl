"""
Subroutine dtron

This subroutine implements a trust region Newton method for the
solution of large bound-constrained optimization problems

  min { f(x) : xl <= x <= xu }

where the Hessian matrix is sparse. The user must evaluate the
function, gradient, and the Hessian matrix.
"""
function dtron(n,x,xl,xu,f,g,a,adiag,acol_ptr,arow_ind,
               frtol,fatol,fmin,cgtol,itermax,delta,task,
               b,bdiag,bcol_ptr,brow_ind,
               l,ldiag,lcol_ptr,lrow_ind,
               xc,s,indfree,
               isave,dsave,wa,iwa)
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

    work = ""

    # Initialization section.

    if unsafe_string(pointer(task), 5) == "START"

        # Initialize local variables.

        iter = 1
        iterscg = 0
        alphac = one
        work = "COMPUTE"

    else

        # Restore local variables.

        if isave[1] == 1
            work = "COMPUTE"
        elseif isave[1] == 2
            work = "EVALUATE"
        elseif isave[1] == 3
            work = "NEWX"
        end
        iter = isave[2]
        iterscg = isave[3]
        fc = dsave[1]
        alphac = dsave[2]
        prered = dsave[3]
    end

    # Search for a lower function value.

    search = true
    while search

        # Compute a step and evaluate the function at the trial point.

        if work == "COMPUTE"

            # Save the best function value, iterate, and gradient.

            fc = f
            dcopy(n,x,1,xc,1)

            # Compute the Cauchy step and store in s.

            alphac = dcauchy(n,x,xl,xu,a,adiag,acol_ptr,arow_ind,g,delta,
                            alphac,s,wa)

            # Compute the projected Newton step.

            info,iters = dspcg(n,x,xl,xu,a,adiag,acol_ptr,arow_ind,g,delta,
                               cgtol,s,5,itermax,
                               b,bdiag,bcol_ptr,brow_ind,
                               l,ldiag,lcol_ptr,lrow_ind,
                               indfree,view(wa,1:n),view(wa,n+1:2*n),
                               view(wa,2*n+1:7*n),iwa)

            # Compute the predicted reduction.

            dssyax(n,a,adiag,acol_ptr,arow_ind,s,wa)
            prered = -(ddot(n,s,1,g,1) + p5*ddot(n,s,1,wa,1))
            iterscg = iterscg + iters

            # Set task to compute the function.

            task[1] = UInt8('F')
        end

        # Evaluate the step and determine if the step is successful.

        if work == "EVALUATE"

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

                task[1] = UInt8('G'); task[2] = UInt8('H')
                iter = iter + 1

            else

                # Unsuccessful iterate.

                task[1] = UInt8('F')
                dcopy(n,xc,1,x,1)
                f = fc

            end

            # Test for convergence.

            if f < fmin
                for (i,s) in enumerate("WARNING: F .LT. FMIN")
                    task[i] = UInt8(s)
                end
            end
            if abs(actred) <= fatol && prered <= fatol
                for (i,s) in enumerate("CONVERGENCE: FATOL TEST SATISFIED")
                    task[i] = UInt8(s)
                end
            end
            if abs(actred) <= frtol*abs(f) && prered <= frtol*abs(f)
                for (i,s) in enumerate("CONVERGENCE: FRTOL TEST SATISFIED")
                    task[i] = UInt8(s)
                end
            end
        end

        # Test for continuation of search

        if Char(task[1]) == 'F' && work == "EVALUATE"
            search = true
            work = "COMPUTE"
        else
            search = false
        end
    end

    if work == "NEWX"
        for (i,s) in enumerate("NEWX")
            task[i] = UInt8(s)
        end
    end

    # Decide on what work to perform on the next iteration.

    if Char(task[1]) == 'F' && work == "COMPUTE"
        work = "EVALUATE"
    elseif Char(task[1]) == 'F' && work == "EVALUATE"
        work = "COMPUTE"
    elseif unsafe_string(pointer(task),2) == "GH"
        work = "NEWX"
    elseif unsafe_string(pointer(task),4) == "NEWX"
        work = "COMPUTE"
    end

    # Save local variables.

    if work == "COMPUTE"
        isave[1] = 1
    elseif work == "EVALUATE"
        isave[1] = 2
    elseif work == "NEWX"
        isave[1] = 3
    end
    isave[2] = iter
    isave[3] = iterscg

    dsave[1] = fc
    dsave[2] = alphac
    dsave[3] = prered

    return delta
end
