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
    @printf("Run dicfs using CPU . . .\n")
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

    @time Threads.@threads for i=1:nblk
        ExaTron.dicfs(n, n^2, tron_A[i], tron_L[i], 5, alpha,
                      view(iwa,:,i), view(wa1,:,i), view(wa2,:,i))
    end

    for i=1:nblk
        L[:,:,i] .= tron_L[i].vals
    end

    @printf("Run dicfs using GPU . . . \n")
    dA = CuArray{Float64}(undef,(n,n,nblk))
    d_out = CuArray{Float64}(undef,(n,n,nblk))
    for i=1:nblk
        copyto!(dA, A)
    end

    @time @cuda threads=(n,n) blocks=nblk shmem=((2*n+2*n^2)*sizeof(Float64)) run_dicfs_gpu(n,alpha,dA,d_out)
    h_L = zeros(n,n,nblk)
    copyto!(h_L, d_out)

    @printf("norm(L(cpu) - L(gpu)): %e\n", maximum(norm.(tril.(eachslice(L,dims=3)) .- tril.(eachslice(h_L,dims=3)))))
    return
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
    @printf("Run spcg using CPU . . .\n")
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
    @printf("   %.6f seconds\n", tcpu.time)

    @printf("Run spcg using GPU . . . \n")

    tgpu = @timed CUDA.@sync @cuda threads=(n,n) blocks=nblk shmem=((3*n)*sizeof(Int)+(12*n+3*(n^2))*sizeof(Float64)) run_spcg_gpu(n,delta,rtol,cg_itermax,dx,dxl,dxu,dA,dg,ds,d_out)
    @printf("   %.6f seconds\n", tgpu.time)

    h_x = zeros(n,nblk)
    copyto!(h_x, d_out)

    @printf("norm(x(cpu) - x(gpu)): %e\n", maximum(norm.(eachslice(x .- h_x,dims=2))))
    return tcpu, tgpu
end

function perf_avg_spcg(;r=10,n=8,nblk=5120)
    @printf("Warming up Julia code . . .\n")
    perf_spcg(;n=n,nblk=nblk)
    @printf("DONE warm-up.\n")

    tcpu_time = 0
    tgpu_time = 0
    i = 0
    for i=1:r
        tcpu, tgpu = perf_spcg(;n=n,nblk=nblk)
        tcpu_time += tcpu.time
        tgpu_time += tgpu.time
    end

    @printf("Average of %d runs:\n", r)
    @printf("   %.5e (CPU)\n", tcpu_time/r)
    @printf("   %.5e (GPU)\n", tgpu_time/r)
    @printf("   %.5e (CPU/GPU)\n", tcpu_time/tgpu_time)

    return
end