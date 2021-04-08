"""
Driver to run TRON on GPU. This should be called from a kernel.
"""
@inline function tron_kernel(n::Int, shift::Int, max_feval::Int, max_minor::Int, gtol::T, scale::T, use_polar::Bool,
                     x::CuDeviceArray{T,1}, xl::CuDeviceArray{T,1},
                     xu::CuDeviceArray{T,1},
                     param::CuDeviceArray{T,2},
                     YffR::T, YffI::T,
                     YftR::T, YftI::T,
                     YttR::T, YttI::T,
                     YtfR::T, YtfI::T) where T
    tx = threadIdx().x

    g = @cuDynamicSharedMem(T, n, (3*n)*sizeof(T))
    xc = @cuDynamicSharedMem(T, n, (4*n)*sizeof(T))
    s = @cuDynamicSharedMem(T, n, (5*n)*sizeof(T))
    wa = @cuDynamicSharedMem(T, n, (6*n)*sizeof(T))
    wa1 = @cuDynamicSharedMem(T, n, (7*n)*sizeof(T))
    wa2 = @cuDynamicSharedMem(T, n, (8*n)*sizeof(T))
    wa3 = @cuDynamicSharedMem(T, n, (9*n)*sizeof(T))
    wa4 = @cuDynamicSharedMem(T, n, (10*n)*sizeof(T))
    wa5 = @cuDynamicSharedMem(T, n, (11*n)*sizeof(T))
    gfree = @cuDynamicSharedMem(T, n, (12*n)*sizeof(T))
    dsave = @cuDynamicSharedMem(T, n, (13*n)*sizeof(T))
    indfree = @cuDynamicSharedMem(Int, n, (14*n)*sizeof(T))
    iwa = @cuDynamicSharedMem(Int, 2*n, n*sizeof(Int) + (14*n)*sizeof(T))
    isave = @cuDynamicSharedMem(Int, n, (3*n)*sizeof(Int) + (14*n)*sizeof(T))

    A = @cuDynamicSharedMem(T, (n,n), (14*n)*sizeof(T)+(4*n)*sizeof(Int))
    B = @cuDynamicSharedMem(T, (n,n), (14*n+n^2)*sizeof(T)+(4*n)*sizeof(Int))
    L = @cuDynamicSharedMem(T, (n,n), (14*n+2*n^2)*sizeof(T)+(4*n)*sizeof(Int))

    if tx <= n
        @inbounds for j=1:n
            A[tx,j] = zero(T)
            B[tx,j] = zero(T)
            L[tx,j] = zero(T)
        end
    end
    CUDA.sync_threads()

    task = 0
    status = 0

    delta = zero(T)
    fatol = zero(T)
    frtol = eps(T)^T(2/3)
    fmin = T(-1e32)
    cgtol = T(0.1)
    cg_itermax = n

    f = zero(T)
    nfev = 0
    ngev = 0
    nhev = 0
    minor_iter = 0
    search = true

    while search

        # [0|1]: Evaluate function.

        if task == 0 || task == 1
            if use_polar
                f = eval_f_polar_kernel(n, shift, scale, x, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI)
            else
                f = eval_f_kernel(n, scale, x, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI)
            end
            nfev += 1
            if nfev >= max_feval
                search = false
            end

        end

        # [2] G or H: Evaluate gradient and Hessian.

        if task == 0 || task == 2
            if use_polar
                eval_grad_f_polar_kernel(n, shift, scale, x, g, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI)
                eval_h_polar_kernel(n, shift, scale, x, A, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI)
            else
                eval_grad_f_kernel(n, scale, x, g, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI)
                eval_h_kernel(n, scale, x, A, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI)
            end
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
    return

end
