using Random
using LinearAlgebra
using CUDA
using Printf
using ExaTron

function run_dicfs_gpu(n::Int, alpha::Float64,
                       dA::CuDeviceArray{Float64},
                       d_out::CuDeviceArray{Float64})
    tx = threadIdx().x
    ty = threadIdx().y
    blk = blockIdx().x

    wa1 = @cuDynamicSharedMem(Float64, n)
    wa2 = @cuDynamicSharedMem(Float64, n, n*sizeof(Float64))
    A = @cuDynamicSharedMem(Float64, (n,n), (2*n)*sizeof(Float64))
    L = @cuDynamicSharedMem(Float64, (n,n), (2*n+n^2)*sizeof(Float64))

    A[tx,ty] = dA[tx,ty,blk]
    CUDA.sync_threads()

    ExaTron.dicfs(n, alpha, A, L, wa1, wa2)
    d_out[tx,ty,blk] = L[tx,ty]
    CUDA.sync_threads()
    return
end

function perf_dicfs(;n=8,nblk=5120)
    L = rand(n,n,nblk)
    A = zeros(n,n,nblk)
    tron_A = Array{ExaTron.TronDenseMatrix}(undef,nblk)
    tron_L = Array{ExaTron.TronDenseMatrix}(undef,nblk)
    for i=1:nblk
        A[:,:,i] .= tril(L[:,:,i])*transpose(L[:,:,i])
        A[:,:,i] .= tril(A[:,:,i]) .+ (transpose(tril(A[:,:,i])) .- Diagonal(A[:,:,i]))
        for j=1:n
            A[j,j,i] = -A[j,j,i]
        end
        tron_A[i] = ExaTron.TronDenseMatrix{Array}(n)
        tron_A[i].vals .= A[:,:,i]
        tron_L[i] = ExaTron.TronDenseMatrix{Array}(n)
    end

    iwa = zeros(Int,3*n,nblk)
    wa1 = zeros(n,nblk)
    wa2 = zeros(n,nblk)
    alpha = 1.0

    tcpu = @timed Threads.@threads for i=1:nblk
            ExaTron.dicfs(n, n^2, tron_A[i], tron_L[i], 5, alpha,
                        view(iwa,:,i), view(wa1,:,i), view(wa2,:,i))
    end

    dA = CuArray{Float64}(undef,(n,n,nblk))
    d_out = CuArray{Float64}(undef,(n,n,nblk))
    for i=1:nblk
        copyto!(dA, A)
    end

    tgpu = @timed CUDA.@sync @cuda threads=(n,n) blocks=nblk shmem=((2*n+2*n^2)*sizeof(Float64)) run_dicfs_gpu(n,alpha,dA,d_out)
    h_L = zeros(n,n,nblk)
    copyto!(h_L, d_out)

    for i=1:nblk
        L[:,:,i] .= tron_L[i].vals
    end
    err = maximum(norm.(tril.(eachslice(L,dims=3)) .- tril.(eachslice(h_L,dims=3))))

    return tcpu, tgpu, err
end

function run_spcg_gpu(n::Int, delta::Float64, rtol::Float64,
                      cg_itermax::Int, dx::CuDeviceArray{Float64},
                      dxl::CuDeviceArray{Float64},
                      dxu::CuDeviceArray{Float64},
                      dA::CuDeviceArray{Float64},
                      dg::CuDeviceArray{Float64},
                      ds::CuDeviceArray{Float64},
                      d_out::CuDeviceArray{Float64})
    tx = threadIdx().x
    ty = threadIdx().y
    blk = blockIdx().x

    x = @cuDynamicSharedMem(Float64, n)
    xl = @cuDynamicSharedMem(Float64, n, n*sizeof(Float64))
    xu = @cuDynamicSharedMem(Float64, n, (2*n)*sizeof(Float64))
    g = @cuDynamicSharedMem(Float64, n, (3*n)*sizeof(Float64))
    s = @cuDynamicSharedMem(Float64, n, (4*n)*sizeof(Float64))
    w = @cuDynamicSharedMem(Float64, n, (5*n)*sizeof(Float64))
    wa1 = @cuDynamicSharedMem(Float64, n, (6*n)*sizeof(Float64))
    wa2 = @cuDynamicSharedMem(Float64, n, (7*n)*sizeof(Float64))
    wa3 = @cuDynamicSharedMem(Float64, n, (8*n)*sizeof(Float64))
    wa4 = @cuDynamicSharedMem(Float64, n, (9*n)*sizeof(Float64))
    wa5 = @cuDynamicSharedMem(Float64, n, (10*n)*sizeof(Float64))
    gfree = @cuDynamicSharedMem(Float64, n, (11*n)*sizeof(Float64))
    indfree = @cuDynamicSharedMem(Int, n, (12*n)*sizeof(Float64))
    iwa = @cuDynamicSharedMem(Int, 2*n, n*sizeof(Int) + (12*n)*sizeof(Float64))

    A = @cuDynamicSharedMem(Float64, (n,n), (12*n)*sizeof(Float64)+(3*n)*sizeof(Int))
    B = @cuDynamicSharedMem(Float64, (n,n), (12*n+n^2)*sizeof(Float64)+(3*n)*sizeof(Int))
    L = @cuDynamicSharedMem(Float64, (n,n), (12*n+2*n^2)*sizeof(Float64)+(3*n)*sizeof(Int))

    A[tx,ty] = dA[tx,ty,blk]
    if ty == 1
        x[tx] = dx[tx,blk]
        xl[tx] = dxl[tx,blk]
        xu[tx] = dxu[tx,blk]
        g[tx] = dg[tx,blk]
        s[tx] = ds[tx,blk]
    end
    CUDA.sync_threads()

    ExaTron.dspcg(n, delta, rtol, cg_itermax, x, xl, xu,
                  A, g, s, B, L, indfree, gfree, w, iwa,
                  wa1, wa2, wa3, wa4, wa5)

    if ty == 1
        d_out[tx,blk] = x[tx]
    end
    CUDA.sync_threads()

    return
end

function perf_spcg(;n=8,nblk=5120)
    L = rand(n,n,nblk)
    A = zeros(n,n,nblk)
    tron_A = Array{ExaTron.TronDenseMatrix}(undef,nblk)
    tron_B = Array{ExaTron.TronDenseMatrix}(undef,nblk)
    tron_L = Array{ExaTron.TronDenseMatrix}(undef,nblk)
    x = rand(n,nblk)
    xl = x .- abs.(rand(n,nblk))
    xu = x .+ abs.(rand(n,nblk))
    g = zeros(n,nblk)
    for i=1:nblk
        A[:,:,i] .= tril(L[:,:,i])*transpose(L[:,:,i])
        A[:,:,i] .= tril(A[:,:,i]) .+ (transpose(tril(A[:,:,i])) .- Diagonal(A[:,:,i]))
        for j=1:n
            A[j,j,i] = -A[j,j,i]
        end
        g[:,i] .= A[:,:,i]*x[:,i] .+ rand(n)
        tron_A[i] = ExaTron.TronDenseMatrix{Array}(n)
        tron_A[i].vals .= A[:,:,i]
        tron_B[i] = ExaTron.TronDenseMatrix{Array}(n)
        tron_L[i] = ExaTron.TronDenseMatrix{Array}(n)
    end

    w = zeros(n,nblk)
    s = rand(n,nblk)
    gfree = zeros(n,nblk)
    indfree = zeros(Int,n,nblk)
    iwa = zeros(Int,3*n,nblk)
    wa = zeros(5*n,nblk)
    cg_itermax = n
    rtol = 1e-6
    delta = 2.0*maximum(norm.(eachslice(g,dims=2)))

    dx = CuArray{Float64}(undef, (n,nblk))
    dxl = CuArray{Float64}(undef, (n,nblk))
    dxu = CuArray{Float64}(undef, (n,nblk))
    dA = CuArray{Float64}(undef, (n,n,nblk))
    dg = CuArray{Float64}(undef, (n,nblk))
    ds = CuArray{Float64}(undef, (n,nblk))
    d_out = CuArray{Float64}(undef, (n,nblk))

    copyto!(dx, x)
    copyto!(dxl, xl)
    copyto!(dxu, xu)
    copyto!(dA, A)
    copyto!(dg, g)
    copyto!(ds, s)

    tcpu = @timed Threads.@threads for i=1:nblk
            ExaTron.dspcg(n, view(x,:,i), view(xl,:,i), view(xu,:,i),
                        tron_A[i], view(g,:,i), delta, rtol, view(s,:,i),
                        5, n, tron_B[i], tron_L[i], view(indfree,:,i),
                        view(gfree,:,i), view(w,:,i), view(wa,:,i),
                        view(iwa,:,i))
    end

    tgpu = @timed CUDA.@sync @cuda threads=(n,n) blocks=nblk shmem=((3*n)*sizeof(Int)+(12*n+3*(n^2))*sizeof(Float64)) run_spcg_gpu(n,delta,rtol,cg_itermax,dx,dxl,dxu,dA,dg,ds,d_out)
    h_x = zeros(n,nblk)
    copyto!(h_x, d_out)
    err = maximum(norm.(eachslice(x .- h_x,dims=2)))

    return tcpu, tgpu, err
end

function run_tron_gpu(n::Int, f::CuDeviceArray{Float64}, frtol::Float64, fatol::Float64, fmin::Float64,
                     cgtol::Float64, cg_itermax::Int, delta::Float64, task::Int,
                     disave::CuDeviceArray{Int}, ddsave::CuDeviceArray{Float64},
                     dx::CuDeviceArray{Float64}, dxl::CuDeviceArray{Float64},
                     dxu::CuDeviceArray{Float64}, dA::CuDeviceArray{Float64},
                     dg::CuDeviceArray{Float64}, d_out::CuDeviceArray{Float64})
    tx = threadIdx().x
    ty = threadIdx().y
    blk = blockIdx().x

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
    indfree = @cuDynamicSharedMem(Int, n, (13*n)*sizeof(Float64))
    iwa = @cuDynamicSharedMem(Int, 2*n, n*sizeof(Int) + (13*n)*sizeof(Float64))

    A = @cuDynamicSharedMem(Float64, (n,n), (13*n)*sizeof(Float64)+(3*n)*sizeof(Int))
    B = @cuDynamicSharedMem(Float64, (n,n), (13*n+n^2)*sizeof(Float64)+(3*n)*sizeof(Int))
    L = @cuDynamicSharedMem(Float64, (n,n), (13*n+2*n^2)*sizeof(Float64)+(3*n)*sizeof(Int))

    fval = f[blk]
    A[tx,ty] = dA[tx,ty,blk]
    if ty == 1
        x[tx] = dx[tx,blk]
        xl[tx] = dxl[tx,blk]
        xu[tx] = dxu[tx,blk]
        g[tx] = dg[tx,blk]
    end
    CUDA.sync_threads()

    ExaTron.dtron(n, x, xl, xu, fval, g, A, frtol, fatol, fmin, cgtol,
        cg_itermax, delta, task, B, L, xc, s, indfree, gfree,
        disave, ddsave, wa, iwa, wa1, wa2, wa3, wa4, wa5)
    if ty == 1
        d_out[tx,blk] = x[tx]
    end
    CUDA.sync_threads()

    return
end

function perf_tron(;n=8,nblk=5120)
    L = rand(n,n,nblk)
    A = zeros(n,n,nblk)
    tron_A = Array{ExaTron.TronDenseMatrix}(undef,nblk)
    tron_B = Array{ExaTron.TronDenseMatrix}(undef,nblk)
    tron_L = Array{ExaTron.TronDenseMatrix}(undef,nblk)
    x = rand(n,nblk)
    xl = x .- abs.(rand(n,nblk))
    xu = x .+ abs.(rand(n,nblk))
    c = rand(n,nblk)
    g = zeros(n,nblk)
    f = zeros(nblk)
    for i=1:nblk
        A[:,:,i] .= tril(L[:,:,i])*transpose(L[:,:,i])
        A[:,:,i] .= tril(A[:,:,i]) .+ (transpose(tril(A[:,:,i])) .- Diagonal(A[:,:,i]))
        for j=1:n
            A[j,j,i] = -A[j,j,i]
        end
        g[:,i] .= A[:,:,i]*x[:,i] .+ c[:,i]
        f[i] = 0.5*transpose(x[:,i])*A[:,:,i]*x[:,i] + transpose(x[:,i])*c[:,i]
        tron_A[i] = ExaTron.TronDenseMatrix{Array}(n)
        tron_A[i].vals .= A[:,:,i]
        tron_B[i] = ExaTron.TronDenseMatrix{Array}(n)
        tron_L[i] = ExaTron.TronDenseMatrix{Array}(n)
    end

    w = zeros(n,nblk)
    xc = zeros(n,nblk)
    s = zeros(n,nblk)
    gfree = zeros(n,nblk)
    indfree = zeros(Int,n,nblk)
    iwa = zeros(Int,3*n,nblk)
    wa = zeros(7*n,nblk)
    isave = zeros(Int,3,nblk)
    dsave = zeros(3,nblk)

    task = 0
    cg_itermax = n
    fatol = 0.0
    frtol = 1e-12
    fmin = -1e32
    cgtol = 0.1
    delta = 2.0*maximum(norm.(eachslice(g,dims=2)))

    df = CuArray{Float64}(undef, nblk)
    dx = CuArray{Float64}(undef, (n,nblk))
    dxl = CuArray{Float64}(undef, (n,nblk))
    dxu = CuArray{Float64}(undef, (n,nblk))
    dA = CuArray{Float64}(undef, (n,n,nblk))
    dg = CuArray{Float64}(undef, (n,nblk))
    disave = CuArray{Int}(undef, (n,nblk))
    ddsave = CuArray{Float64}(undef, (n,nblk))
    d_out = CuArray{Float64}(undef, (n,nblk))

    copyto!(df, f)
    copyto!(dx, x)
    copyto!(dxl, xl)
    copyto!(dxu, xu)
    copyto!(dA, A)
    copyto!(dg, g)

    task_str = zeros(UInt8,60,nblk)
    for (i,s) in enumerate("START")
        task_str[i,:] .= UInt8(s)
    end

    tcpu = @timed Threads.@threads for i=1:nblk
            ExaTron.dtron(n, view(x,:,i), view(xl,:,i), view(xu,:,i),
                          f[i], view(g,:,i), tron_A[i], frtol, fatol, fmin, cgtol,
                          cg_itermax, delta, view(task_str,:,i), tron_B[i], tron_L[i],
                          view(xc,:,i), view(s,:,i), view(indfree,:,i),
                          view(isave,:,i), view(dsave,:,i), view(wa,:,i), view(iwa,:,i))
    end

    tgpu = @timed CUDA.@sync @cuda threads=(n,n) blocks=nblk shmem=((3*n)*sizeof(Int)+(13*n+3*(n^2))*sizeof(Float64)) run_tron_gpu(n,df,frtol,fatol,fmin,cgtol,cg_itermax,delta,task,disave,ddsave,dx,dxl,dxu,dA,dg,d_out)
    h_x = zeros(n,nblk)
    copyto!(h_x, d_out)
    err = maximum(norm.(eachslice(x .- h_x,dims=2)))

    return tcpu, tgpu, err
end

function perf_avg(fname="perf_tron";r=10,n=8,nblk=5120,log=true)
    @printf("Warming up Julia code . . . ")
    flush(stdout)
    f = Meta.parse(fname*"(;n=$n,nblk=$nblk)")
    eval(f);
    @printf("DONE\n")

    @printf("Run %s() . . .\n", fname)
    f = Meta.parse(fname*"(;n=$n,nblk=$nblk)")
    tcpu_time = 0
    tgpu_time = 0

    if log
        @printf("\n%9s\t%9s\t%9s\t%8s\n", "Iteration", "CPU(s)", "GPU(s)", "Error")
    end

    for i=1:r
        tcpu, tgpu, err = eval(f)
        if log
            @printf("%9d\t%5.5e\t%5.5e\t%5.5e\n", i, tcpu.time, tgpu.time, err)
        end
        tcpu_time += tcpu.time
        tgpu_time += tgpu.time
    end

    @printf("\n ** Statistics:\n\n")
    @printf("Number of iterations . . . %5d\n", r)
    @printf("Average(CPU) . . . . . . . %.5e\n", tcpu_time/r)
    @printf("Average(GPU) . . . . . . . %.5e\n", tgpu_time/r)
    @printf("Average ratio(CPU/GPU) . . %.5e\n", tcpu_time/tgpu_time)

    return
end

