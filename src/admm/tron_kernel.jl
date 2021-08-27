"""
Driver to run TRON on GPU. This should be called from a kernel.
"""
@inline function tron_kernel(n::Int, shift::Int, max_feval::Int, max_minor::Int, gtol::Float64, scale::Float64, use_polar::Bool,
                     x, xl,
                     xu,
                     param,
                     YffR::Float64, YffI::Float64,
                     YftR::Float64, YftI::Float64,
                     YttR::Float64, YttI::Float64,
                     YtfR::Float64, YtfI::Float64,
                     I, J
                     )

    tx = J
    g = @localmem Float64 (4,)
    xc = @localmem Float64 (4,)
    s = @localmem Float64 (4,)
    wa = @localmem Float64 (4,)
    wa1 = @localmem Float64 (4,)
    wa2 = @localmem Float64 (4,)
    wa3 = @localmem Float64 (4,)
    wa4 = @localmem Float64 (4,)
    wa5 = @localmem Float64 (4,)
    wa = @localmem Float64 (4,)
    gfree = @localmem Float64 (4,)
    dsave = @localmem Float64 (4,)
    indfree = @localmem Int (4,)
    iwa = @localmem Int (4,)
    isave = @localmem Int (4,)

    A = @localmem Float64 (4,4)
    B = @localmem Float64 (4,4)
    L = @localmem Float64 (4,4)

    if tx <= n
        @inbounds for j=1:n
            A[tx,j] = 0.0
            B[tx,j] = 0.0
            L[tx,j] = 0.0
        end
    end
    @synchronize

    task = 0
    status = 0

    delta = 0.0
    fatol = 0.0
    frtol = 1e-12
    fmin = -1e32
    cgtol = 0.1
    cg_itermax = n

    f = 0.0
    nfev = 0
    ngev = 0
    nhev = 0
    minor_iter = 0
    search = true

    while search

        # [0|1]: Evaluate function.

        if task == 0 || task == 1
            if use_polar
                f = eval_f_polar_kernel(n, shift, scale, x, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI, I, J)
            else
                f = eval_f_kernel(n, scale, x, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI, I, J)
            end
            nfev += 1
            if nfev >= max_feval
                search = false
            end

        end

        # [2] G or H: Evaluate gradient and Hessian.

        if task == 0 || task == 2
            if use_polar
                eval_grad_f_polar_kernel(n, shift, scale, x, g, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI, I, J)
                eval_h_polar_kernel(n, shift, scale, x, A, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI, I, J)
            else
                eval_grad_f_kernel(n, scale, x, g, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI, I, J)
                eval_h_kernel(n, scale, x, A, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI, I, J)
            end
            ngev += 1
            nhev += 1
            minor_iter += 1
        end

        # Initialize the trust region bound.

        if task == 0
            gnorm0 = dnrm2(n, g, 1, I, J)
            delta = gnorm0
        end

        # Call Tron.

        if search
            delta, task = dtron(n, x, xl, xu, f, g, A, frtol, fatol, fmin, cgtol,
                                cg_itermax, delta, task, B, L, xc, s, indfree, gfree,
                                isave, dsave, wa, iwa, wa1, wa2, wa3, wa4, wa5, I, J)
        end

        # [3] NEWX: a new point was computed.

        if task == 3
            gnorm_inf = dgpnorm(n, x, xl, xu, g, I, J)

            if gnorm_inf <= gtol
                task = 4
            end

            if minor_iter >= max_minor
                status = 1
                search = false
            end
        end

        # [4] CONV: convergence was achieved.
        # [10] : warning fval is less than fmin

        if task == 4 || task == 10
            search = false
        end
    end

    @synchronize

    return status, minor_iter
end
