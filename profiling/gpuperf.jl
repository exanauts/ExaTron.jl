using Random
using LinearAlgebra
using CUDA
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
    println("Run dicfs using CPU . . .")
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

    iwa = zeros(Int, 3*n,nblk)
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

    println("Run dicfs using GPU . . . ")
    dA = CuArray{Float64}(undef,(n,n,nblk))
    d_out = CuArray{Float64}(undef,(n,n,nblk))
    for i=1:nblk
        copyto!(dA, A)
    end

    @time @cuda threads=(n,n) blocks=nblk shmem=((2*n+2*n^2)*sizeof(Float64)) run_dicfs_gpu(n,alpha,dA,d_out)
    h_L = zeros(n,n,nblk)
    copyto!(h_L, d_out)

    println("norm(L(cpu) - L(gpu)): ", maximum(norm.(tril.(eachslice(L,dims=3)) .- tril.(eachslice(h_L,dims=3)))))
    return
end