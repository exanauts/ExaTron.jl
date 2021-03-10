"""
Driver to run TRON on GPU. This should be called from a kernel.
"""
function tron_kernel(n::Int, max_feval::Int, max_minor::Int, gtol::Float64,
                     x::CuDeviceArray{Float64}, xl::CuDeviceArray{Float64},
                     xu::CuDeviceArray{Float64},
                     param::CuDeviceArray{Float64},
                     YffR::Float64, YffI::Float64,
                     YftR::Float64, YftI::Float64,
                     YttR::Float64, YttI::Float64,
                     YtfR::Float64, YtfI::Float64)
                     #=
                     YffR::CuDeviceArray{Float64}, YffI::CuDeviceArray{Float64},
                     YftR::CuDeviceArray{Float64}, YftI::CuDeviceArray{Float64},
                     YttR::CuDeviceArray{Float64}, YttI::CuDeviceArray{Float64},
                     YtfR::CuDeviceArray{Float64}, YtfI::CuDeviceArray{Float64})
                     =#
    tx = threadIdx().x
    ty = threadIdx().y

    g = @cuDynamicSharedMem(Float64, n, (3*n)*sizeof(Float64))
    xc = @cuDynamicSharedMem(Float64, n, (4*n)*sizeof(Float64))
    s = @cuDynamicSharedMem(Float64, n, (5*n)*sizeof(Float64))
    wa = @cuDynamicSharedMem(Float64, n, (6*n)*sizeof(Float64))
    wa1 = @cuDynamicSharedMem(Float64, n, (7*n)*sizeof(Float64))
    wa2 = @cuDynamicSharedMem(Float64, n, (8*n)*sizeof(Float64))
    wa3 = @cuDynamicSharedMem(Float64, n, (9*n)*sizeof(Float64))
    wa4 = @cuDynamicSharedMem(Float64, n, (10*n)*sizeof(Float64))
    wa5 = @cuDynamicSharedMem(Float64, n, (11*n)*sizeof(Float64))
    gfree = @cuDynamicSharedMem(Float64, n, (12*n)*sizeof(Float64))
    dsave = @cuDynamicSharedMem(Float64, n, (13*n)*sizeof(Float64))
    indfree = @cuDynamicSharedMem(Int, n, (14*n)*sizeof(Float64))
    iwa = @cuDynamicSharedMem(Int, 2*n, n*sizeof(Int) + (14*n)*sizeof(Float64))
    isave = @cuDynamicSharedMem(Int, n, (3*n)*sizeof(Int) + (14*n)*sizeof(Float64))

    A = @cuDynamicSharedMem(Float64, n*n, (14*n)*sizeof(Float64)+(4*n)*sizeof(Int))
    B = @cuDynamicSharedMem(Float64, n*n, (14*n+n^2)*sizeof(Float64)+(4*n)*sizeof(Int))
    L = @cuDynamicSharedMem(Float64, n*n, (14*n+2*n^2)*sizeof(Float64)+(4*n)*sizeof(Int))

    A[n*(ty-1) + tx] = 0.0
    B[n*(ty-1) + tx] = 0.0
    L[n*(ty-1) + tx] = 0.0

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
            f = eval_f_kernel(n, x, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI)
            nfev += 1
            if nfev >= max_feval
                search = false
            end

        end

        # [2] G or H: Evaluate gradient and Hessian.

        if task == 0 || task == 2
            eval_grad_f_kernel(n, x, g, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI)
            eval_h_kernel(n, x, A, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI)
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
            delta, task = dtron(n, x, xl, xu, f, g, A, frtol, fatol, fmin, cgtol,
                                cg_itermax, delta, task, B, L, xc, s, indfree, gfree,
                                isave, dsave, wa, iwa, wa1, wa2, wa3, wa4, wa5)
        end

        # [3] NEWX: a new point was computed.

        if task == 3
            gnorm_inf = dgpnorm(n, x, xl, xu, g)

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

#=
function tron_kernel(n, max_feval, max_minor, x, xl, xu)

    # We start with a shared memory allocation.
    # The first 3*n*sizeof(Float64) bytes are used for x, xl, and xu.

    g = @cuDynamicSharedMem(Float64, n, (3*n)*sizeof(Float64))
    xc = @cuDynamicSharedMem(Float64, n, (4*n)*sizeof(Float64))
    s = @cuDynamicSharedMem(Float64, n, (5*n)*sizeof(Float64))
    wa = @cuDynamicSharedMem(Float64, n, (6*n)*sizeof(Float64))
    wa1 = @cuDynamicSharedMem(Float64, n, (7*n)*sizeof(Float64))
    wa2 = @cuDynamicSharedMem(Float64, n, (8*n)*sizeof(Float64))
    wa3 = @cuDynamicSharedMem(Float64, n, (9*n)*sizeof(Float64))
    wa4 = @cuDynamicSharedMem(Float64, n, (10*n)*sizeof(Float64))
    wa5 = @cuDynamicSharedMem(Float64, n, (11*n)*sizeof(Float64))
    gfree = @cuDynamicSharedMem(Float64, n, (12*n)*sizeof(Float64))
    dsave = @cuDynamicSharedMem(Float64, 3, (13*n)*sizeof(Float64))
    indfree = @cuDynamicSharedMem(Int, n, (13*n+3)*sizeof(Float64))
    iwa = @cuDynamicSharedMem(Int, 2*n, n*sizeof(Int) + (13*n+3)*sizeof(Float64))
    isave = @cuDynamicSharedMem(Int, 3, (3*n)*sizeof(Int) + (13*n+3)*sizeof(Float64))

    A = @cuDynamicSharedMem(Float64, (n,n), (13*n+3)*sizeof(Float64)+(3*n+3)*sizeof(Int))
    B = @cuDynamicSharedMem(Float64, (n,n), (13*n+3+n^2)*sizeof(Float64)+(3*n+3)*sizeof(Int))
    L = @cuDynamicSharedMem(Float64, (n,n), (13*n+3+2*n^2)*sizeof(Float64)+(3*n+3)*sizeof(Int))

    task = 0
    status = 0

    fatol = 0.0
    frtol = 1e-12
    fmin = -1e32
    cgtol = 0.1
    cg_itermax = n

    f = 0
    nfev = 0
    ngev = 0
    nhev = 0
    minor_iter = 0
    delta = 0.0
    search = true

    while search

        # [0|1]: Evaluate function.

        if task == 0 || task == 1
            f = eval_f(x)
            nfev += 1
            if nfev >= max_feval
                search = false
            end
        end

        # [2] G or H: Evaluate gradient and Hessian.

        if task == 2
            eval_g(x, g)
            eval_h(1.0, x, A)
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
            delta, task = dtron(n, x, xl, xu, f, g, A, frtol, fatol, fmin, cgtol,
                                cg_itermax, delta, task, B, L, xc, s, indfree, gfree,
                                isave, dsave, wa, iwa, wa1, wa2, wa3, wa4, wa5)
        end

        # [3] NEWX: a new point was computed.

        if task == 3
            gnorm_inf = dgpnorm(n, x, xl, xu, g)
            if gnorm_inf <= gtol
                task = 4
            end

            if minor_iter >= max_minor
                status = 1
                search = false
            end
        end

        # [4] CONV: convergence was achieved.

        if task == 4
            search = false
        end
    end

    return status
end
=#