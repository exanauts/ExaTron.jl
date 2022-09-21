# Assume that scaling factor has already been applied to Q and c.
@inline function eval_qp_f_kernel(n::Int, x, Q, c, tx)
    # f = xQx/2 + cx
    f = 0.0
    @inbounds begin
        for j=1:n
            for i=1:n
                f += x[i]*Q[i,j]*x[j]
            end
        end
        f *= 0.5
        for j=1:n
            f += c[j]*x[j]
        end
    end
    return f
end

@inline function eval_qp_grad_f_kernel(n::Int, x, g, Q, c, tx)
    # g = Qx + c

    @inbounds begin
        if tx <= n
            g[tx] = c[tx]
        end
        @synchronize
        if tx <= n
            for j=1:n
                g[tx] += Q[tx,j]*x[j]
            end
        end
        @synchronize
    end
    return
end

@inline function ExaTron.tron_qp_kernel(n::Int, max_feval::Int, max_minor::Int, gtol::Float64, scale::Float64,
    x, xl, xu,
    A, c, tx, I)

    g = @localmem Float64, (n,)
    xc = @localmem Float64, (n,)
    s = @localmem Float64, (n,)
    wa = @localmem Float64, (n,)
    wa1 = @localmem Float64, (n,)
    wa2 = @localmem Float64, (n,)
    wa3 = @localmem Float64, (n,)
    wa4 = @localmem Float64, (n,)
    wa5 = @localmem Float64, (n,)
    gfree = @localmem Float64, (n,)
    dsave = @localmem Float64, (3,)
    indfree = @localmem Int, (n,)
    iwa = @localmem Int, (2*n,)
    isave = @localmem Int, (3,)

    B = @localmem Float64 (n,n)
    L = @localmem Float64 (n,n)

    if tx <= n
        @inbounds begin
            for j=1:n
                B[tx,j] = 0.0
                L[tx,j] = 0.0
            end
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
            f = eval_qp_f_kernel(n, x, A, c, tx)
            nfev += 1
            if nfev >= max_feval
                search = false
            end
        end

        # [2] G or H: Evaluate gradient and Hessian.

        if task == 0 || task == 2
            eval_qp_grad_f_kernel(n, x, g, A, c, tx)
            # We do not have to evaluate Hessian since A does not change.
            ngev += 1
            nhev += 1
            minor_iter += 1
        end

        # Initialize the trust region bound.

        if task == 0
            gnorm0 = dnrm2(n, g, 1, tx)
            delta = gnorm0
        end

        # Call Tron.

        if search
            delta, task = ExaTron.dtron(n, x, xl, xu, f, g, A, frtol, fatol, fmin, cgtol,
                                cg_itermax, delta, task, B, L, xc, s, indfree, gfree,
                                isave, dsave, wa, iwa, wa1, wa2, wa3, wa4, wa5, tx)
        end

        # [3] NEWX: a new point was computed.

        if task == 3
            gnorm_inf = ExaTron.dgpnorm(n, x, xl, xu, g, tx)

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
