"""
Driver to run TRON on GPU. This should be called from a kernel.
"""

#=
function tron2_kernel(n::Int, max_feval::Int, max_minor::Int, gtol::Float64,
                      _x::CuDeviceArray{Float64,2}, _xl::CuDeviceArray{Float64,2},
                      _xu::CuDeviceArray{Float64,2})
=#
function tron2_kernel(n::Int, max_feval::Int, max_minor::Int, gtol::Float64,
                      memVecFloat::CuDeviceArray{Float64,2},
                      memVecInt::CuDeviceArray{Int,2},
                      memMatA::CuDeviceArray{Float64,3},
                      memMatB::CuDeviceArray{Float64,3},
                      memMatL::CuDeviceArray{Float64,3})

    tx = threadIdx().x
    I = blockIdx().x

    x = @view(memVecFloat[1:n, I])
    xl = @view(memVecFloat[n+1:2*n, I])
    xu = @view(memVecFloat[2*n+1:3*n, I])
    g = @view(memVecFloat[3*n+1:4*n, I])
    xc = @view(memVecFloat[4*n+1:5*n, I])
    s = @view(memVecFloat[5*n+1:6*n, I])
    wa = @view(memVecFloat[6*n+1:7*n, I])
    wa1 = @view(memVecFloat[7*n+1:8*n, I])
    wa2 = @view(memVecFloat[8*n+1:9*n, I])
    wa3 = @view(memVecFloat[9*n+1:10*n, I])
    wa4 = @view(memVecFloat[10*n+1:11*n, I])
    wa5 = @view(memVecFloat[11*n+1:12*n, I])
    gfree = @view(memVecFloat[12*n+1:13*n, I])
    dsave = @view(memVecFloat[13*n+1:14*n, I])
    indfree = @view(memVecInt[1:n, I])
    iwa = @view(memVecInt[n+1:3*n, I])
    isave = @view(memVecInt[3*n+1:4*n, I])
    A = @view(memMatA[:,:,I])
    B = @view(memMatB[:,:,I])
    L = @view(memMatL[:,:,I])

    #=
    if tx == 1
        @cuprintln("I = ", I, " xl[1] = ", xl[1], " memVecFloat[2,I] = ", memVecFloat[2,I])
    end
    =#

    CUDA.sync_threads()

    #=
    x = @cuDynamicSharedMem(Float64, n)
    xl = @cuDynamicSharedMem(Float64, n, n*sizeof(Float64))
    xu = @cuDynamicSharedMem(Float64, n, (2*n)*sizeof(Float64))
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

    A = @cuDynamicSharedMem(Float64, (n,n), (14*n)*sizeof(Float64)+(4*n)*sizeof(Int))
    B = @cuDynamicSharedMem(Float64, (n,n), (14*n+n^2)*sizeof(Float64)+(4*n)*sizeof(Int))
    L = @cuDynamicSharedMem(Float64, (n,n), (14*n+2*n^2)*sizeof(Float64)+(4*n)*sizeof(Int))

    if tx <= n
        x[tx] = _x[tx,I]
        xl[tx] = _xl[tx,I]
        xu[tx] = _xu[tx,I]

        @inbounds for j=1:n
            A[tx,j] = 0.0
            B[tx,j] = 0.0
            L[tx,j] = 0.0
        end
    end
    CUDA.sync_threads()
    =#

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
            f = eval_f_kernel(n, x)
            nfev += 1
            if nfev >= max_feval
                search = false
            end

        end

        # [2] G or H: Evaluate gradient and Hessian.

        if task == 0 || task == 2
            eval_grad_f_kernel(n, x, g)
            eval_h_kernel(n, x, A)
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

    #=
    if tx == 1
        @cuprintln("I = ", I, " x[1] = ", x[1], " xl[1] = ", xl[1], " xu[1] = ", xu[1],
                   " memVecFloat[1,I] = ", memVecFloat[1,I],
                   " memVecFloat[2,I] = ", memVecFloat[2,I],
                   " memVecFloat[3,I] = ", memVecFloat[3,I])
    end
    =#
    #=
    if tx <= n
        _x[tx,I] = x[tx]
    end
    CUDA.sync_threads()
    =#

    return
end
