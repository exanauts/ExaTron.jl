# Assume that scaling factor has already been applied to Q and c.
@inline function eval_qp_f_kernel(n::Int, x::CuDeviceArray{Float64,1}, Q::CuDeviceArray{Float64,2}, c::CuDeviceArray{Float64,1})
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

@inline function eval_qp_grad_f_kernel(n::Int, x::CuDeviceArray{Float64,1}, g::CuDeviceArray{Float64,1}, Q::CuDeviceArray{Float64,2}, c::CuDeviceArray{Float64,1})
    # g = Qx + c
    tx = threadIdx().x

    @inbounds begin
        if tx <= n
            g[tx] = c[tx]
        end
        CUDA.sync_threads()
        if tx <= n
            for j=1:n
                g[tx] += Q[tx,j]*x[j]
            end
        end
        CUDA.sync_threads()
    end
    return
end

@inline function tron_qp_kernel(n::Int, max_feval::Int, max_minor::Int, gtol::Float64, scale::Float64,
    x::CuDeviceArray{Float64,1}, xl::CuDeviceArray{Float64,1}, xu::CuDeviceArray{Float64,1},
    A::CuDeviceArray{Float64,2}, c::CuDeviceArray{Float64,1})

    tx = threadIdx().x
    I = blockIdx().x

    g = CuDynamicSharedArray(Float64, n, (3*n)*sizeof(Float64))
    xc = CuDynamicSharedArray(Float64, n, (4*n)*sizeof(Float64))
    s = CuDynamicSharedArray(Float64, n, (5*n)*sizeof(Float64))
    wa = CuDynamicSharedArray(Float64, n, (6*n)*sizeof(Float64))
    wa1 = CuDynamicSharedArray(Float64, n, (7*n)*sizeof(Float64))
    wa2 = CuDynamicSharedArray(Float64, n, (8*n)*sizeof(Float64))
    wa3 = CuDynamicSharedArray(Float64, n, (9*n)*sizeof(Float64))
    wa4 = CuDynamicSharedArray(Float64, n, (10*n)*sizeof(Float64))
    wa5 = CuDynamicSharedArray(Float64, n, (11*n)*sizeof(Float64))
    gfree = CuDynamicSharedArray(Float64, n, (12*n)*sizeof(Float64))
    dsave = CuDynamicSharedArray(Float64, 3, (13*n)*sizeof(Float64))
    indfree = CuDynamicSharedArray(Int, n, (13*n+3)*sizeof(Float64))
    iwa = CuDynamicSharedArray(Int, 2*n, n*sizeof(Int) + (13*n+3)*sizeof(Float64))
    isave = CuDynamicSharedArray(Int, 3, (3*n)*sizeof(Int) + (13*n+3)*sizeof(Float64))

    B = CuDynamicSharedArray(Float64, (n,n), (13*n+3+n^2)*sizeof(Float64)+(3*n+3)*sizeof(Int))
    L = CuDynamicSharedArray(Float64, (n,n), (13*n+3+2*n^2)*sizeof(Float64)+(3*n+3)*sizeof(Int))

    if tx <= n
        @inbounds begin
            for j=1:n
                B[tx,j] = 0.0
                L[tx,j] = 0.0
            end
        end
    end
    CUDA.sync_threads()

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
            f = eval_qp_f_kernel(n, x, A, c)
            nfev += 1
            if nfev >= max_feval
                search = false
            end
        end

        # [2] G or H: Evaluate gradient and Hessian.

        if task == 0 || task == 2
            eval_qp_grad_f_kernel(n, x, g, A, c)
            # We do not have to evaluate Hessian since A does not change.
            ngev += 1
            nhev += 1
            minor_iter += 1
        end

        # Initialize the trust region bound.

        if task == 0
            gnorm0 = dnrm2(n, g, 1)
            delta = gnorm0
        end

        # Call Tron.

        if search
            delta, task = ExaTron.dtron(n, x, xl, xu, f, g, A, frtol, fatol, fmin, cgtol,
                                cg_itermax, delta, task, B, L, xc, s, indfree, gfree,
                                isave, dsave, wa, iwa, wa1, wa2, wa3, wa4, wa5)
        end

        # [3] NEWX: a new point was computed.

        if task == 3
            gnorm_inf = ExaTron.dgpnorm(n, x, xl, xu, g)

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

    CUDA.sync_threads()

    return status, minor_iter
end
