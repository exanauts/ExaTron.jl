using CUDA
using ExaTron
using LinearAlgebra
using Random
using Test

try
    tmp = CuArray{Float64}(undef, 10)
catch e
    throw(e)
end

"""
Test ExaTron's internal routines written for GPU.

The current implementation assumes the following:
  - A thread block takes a matrix structure, (tx,ty), with
    n <= blockDim().x = blockDim().y <= 32 and blockDim().x is even.
  - Arguments passed on to these routines are assumed to be
    of size at least n. This is to prevent multiple thread
    divergence when we call a function with n_hat < n.
    Such a case occurs when we fix active variables.

We test the following routines, where [O] indicates if the routine
is checked if n < blockDim().x is OK.
  - dicf     [O][O]: this routine also tests dnsol and dtsol.
  - dicfs    [O][T]
  - dcauchy  [O][T]
  - dtrpcg   [O][T]
  - dprsrch  [O][T]
  - daxpy    [O][O]
  - dssyax   [O][O]: we do shuffle using blockDim().x.
  - dmid     [O][O]: we could use a single thread only to multiple divergences.
  - dgpstep  [O][O]
  - dbreakpt [O][O]: we use the existing ExaTron implementation as is.
  - dnrm2    [O][O]: we do shuffle using blockDim().x.
  - nrm2     [O][O]: we do shuffle using blockDim().x.
  - dcopy    [O][O]
  - ddot     [O][O]
  - dscal    [O][O]
  - dtrqsol  [O][O]
  - dspcg    [O][T]: we use a single thread to avoid multiple divergences.
  - dgpnorm  [O][O]
  - dtron    [O]
  - driver_kernel
"""

Random.seed!(0)

@testset "CUDA.jl" begin
    itermax = 10
    n = 8
    nblk = 5120

    @testset "dicf" begin
        function dicf_test(n::Int, d_in::CuDeviceArray{Float64},
                            d_out::CuDeviceArray{Float64})
            tx = CUDA.threadIdx().x
            bx = CUDA.blockIdx().x

            L = CuDynamicSharedArray(Float64, (n,n))
            for i in 1:n
                L[i,tx] = d_in[i,tx]
            end
            CUDA.sync_threads()

            # Test Cholesky factorization.
            ExaTron.dicf(n,L)

            if bx == 1
                for i in 1:n
                    d_out[i,tx] = L[i,tx]
                end
            end
            CUDA.sync_threads()
            return
        end

        for i=1:itermax
            L = tril(rand(n,n))
            A = L*transpose(L)
            A .= tril(A) .+ (transpose(tril(A)) .- Diagonal(A))
            tron_A = ExaTron.TronDenseMatrix{Array{Float64,2}}(n)
            tron_A.vals .= A

            d_in = CuArray{Float64,2}(undef, (n,n))
            d_out = CuArray{Float64,2}(undef, (n,n))
            copyto!(d_in, tron_A.vals)
            CUDA.@sync @cuda threads=n blocks=nblk shmem=(n^2*sizeof(Float64)) dicf_test(n,d_in,d_out)
            h_L = zeros(n,n)
            copyto!(h_L, d_out)

            tron_L = ExaTron.TronDenseMatrix{Array{Float64,2}}(n)
            tron_L.vals .= tron_A.vals
            indr = zeros(Int, n)
            indf = zeros(n)
            list = zeros(n)
            w = zeros(n)
            ExaTron.dicf(n, n^2, tron_L, 5, indr, indf, list, w)

            @test norm(tron_A.vals .- tril(h_L)*transpose(tril(h_L))) <= 1e-10
            @test norm(tril(h_L) .- transpose(triu(h_L))) <= 1e-10
            @test norm(tril(tron_L.vals) .- tril(h_L)) <= 1e-10
        end
    end

    @testset "dicfs" begin
        function dicfs_test(n::Int, alpha::Float64,
                            dA::CuDeviceArray{Float64},
                            d_out::CuDeviceArray{Float64})
            tx = CUDA.threadIdx().x
            bx = CUDA.blockIdx().x

            wa1 = CuDynamicSharedArray(Float64, n)
            wa2 = CuDynamicSharedArray(Float64, n, n*sizeof(Float64))
            A = CuDynamicSharedArray(Float64, (n,n), (2*n)*sizeof(Float64))
            L = CuDynamicSharedArray(Float64, (n,n), (2*n+n^2)*sizeof(Float64))

            @inbounds for j=1:n
                A[j,tx] = dA[j,tx]
            end
            CUDA.sync_threads()

            ExaTron.dicfs(n, alpha, A, L, wa1, wa2)
            if bx == 1
                @inbounds for j=1:n
                    d_out[j,tx] = L[j,tx]
                end
            end
            CUDA.sync_threads()

            return
        end

        for i=1:itermax
            L = tril(rand(n,n))
            A = L*transpose(L)
            A .= tril(A) .+ (transpose(tril(A)) .- Diagonal(A))
            tron_A = ExaTron.TronDenseMatrix{Array{Float64,2}}(n)
            tron_L = ExaTron.TronDenseMatrix{Array{Float64,2}}(n)
            tron_A.vals .= A

            dA = CuArray{Float64,2}(undef, (n,n))
            d_out = CuArray{Float64,2}(undef, (n,n))
            alpha = 1.0
            copyto!(dA, tron_A.vals)
            CUDA.@sync @cuda threads=n blocks=nblk shmem=((2*n+2*n^2)*sizeof(Float64)) dicfs_test(n,alpha,dA,d_out)
            h_L = zeros(n,n)
            copyto!(h_L, d_out)
            iwa = zeros(Int, 3*n)
            wa1 = zeros(n)
            wa2 = zeros(n)
            ExaTron.dicfs(n, n^2, tron_A, tron_L, 5, alpha, iwa, wa1, wa2)

            @test norm(tril(h_L) .- transpose(triu(h_L))) <= 1e-10
            @test norm(tril(tron_L.vals) .- tril(h_L)) <= 1e-9

            # Make it negative definite.
            for j=1:n
                tron_A.vals[j,j] = -tron_A.vals[j,j]
            end
            copyto!(dA, tron_A.vals)
            CUDA.@sync @cuda threads=n blocks=nblk shmem=((2*n+2*n^2)*sizeof(Float64)) dicfs_test(n,alpha,dA,d_out)
            copyto!(h_L, d_out)
            ExaTron.dicfs(n, n^2, tron_A, tron_L, 5, alpha, iwa, wa1, wa2)

            @test norm(tril(h_L) .- transpose(triu(h_L))) <= 1e-10
            @test norm(tril(tron_L.vals) .- tril(h_L)) <= 1e-10
        end
    end

    @testset "dcauchy" begin
        function dcauchy_test(n::Int,dx::CuDeviceArray{Float64},
                                dl::CuDeviceArray{Float64},
                                du::CuDeviceArray{Float64},
                                dA::CuDeviceArray{Float64},
                                dg::CuDeviceArray{Float64},
                                delta::Float64,
                                alpha::Float64,
                                d_out1::CuDeviceArray{Float64},
                                d_out2::CuDeviceArray{Float64})
            tx = CUDA.threadIdx().x
            bx = CUDA.blockIdx().x

            x = CuDynamicSharedArray(Float64, n)
            xl = CuDynamicSharedArray(Float64, n, n*sizeof(Float64))
            xu = CuDynamicSharedArray(Float64, n, (2*n)*sizeof(Float64))
            g =  CuDynamicSharedArray(Float64, n, (3*n)*sizeof(Float64))
            s =  CuDynamicSharedArray(Float64, n, (4*n)*sizeof(Float64))
            wa = CuDynamicSharedArray(Float64, n, (5*n)*sizeof(Float64))
            A =  CuDynamicSharedArray(Float64, (n,n), (6*n)*sizeof(Float64))

            for i in 1:n
                A[i,tx] = dA[i,tx]
            end
            x[tx] = dx[tx]
            xl[tx] = dl[tx]
            xu[tx] = du[tx]
            g[tx] = dg[tx]

            alpha = ExaTron.dcauchy(n,x,xl,xu,A,g,delta,alpha,s,wa)
            if bx == 1
                d_out1[tx] = s[tx]
                d_out2[tx] = alpha
            end
            CUDA.sync_threads()

            return
        end

        for i=1:itermax
            L = tril(rand(n,n))
            A = ExaTron.TronDenseMatrix{Array{Float64,2}}(n)
            A.vals .= L*transpose(L)
            A.vals .= tril(A.vals) .+ (transpose(tril(A.vals)) .- Diagonal(A.vals))
            x = rand(n)
            xl = x .- abs.(rand(n))
            xu = x .+ abs.(rand(n))
            g = A.vals*x .+ rand(n)
            s = zeros(n)
            wa = zeros(n)
            alpha = 1.0
            delta = 2.0*norm(g)

            dx = CuArray{Float64}(undef, n)
            dl = CuArray{Float64}(undef, n)
            du = CuArray{Float64}(undef, n)
            dg = CuArray{Float64}(undef, n)
            dA = CuArray{Float64,2}(undef, (n,n))
            d_out1 = CuArray{Float64}(undef, n)
            d_out2 = CuArray{Float64}(undef, n)
            copyto!(dx, x)
            copyto!(dl, xl)
            copyto!(du, xu)
            copyto!(dg, g)
            copyto!(dA, A.vals)
            CUDA.@sync @cuda threads=n blocks=nblk shmem=((6*n+n^2)*sizeof(Float64)) dcauchy_test(n,dx,dl,du,dA,dg,delta,alpha,d_out1,d_out2)
            h_s = zeros(n)
            h_alpha = zeros(n)
            copyto!(h_s, d_out1)
            copyto!(h_alpha, d_out2)

            alpha = ExaTron.dcauchy(n, x, xl, xu, A, g, delta, alpha, s, wa)

            @test norm(s .- h_s) <= 1e-10
            @test norm(alpha .- h_alpha) <= 1e-10
        end
    end

    @testset "dtrpcg" begin
        function dtrpcg_test(n::Int, delta::Float64, tol::Float64,
                                stol::Float64, d_in::CuDeviceArray{Float64},
                                d_g::CuDeviceArray{Float64},
                                d_out_L::CuDeviceArray{Float64},
                                d_out::CuDeviceArray{Float64})
            tx = CUDA.threadIdx().x
            bx = CUDA.blockIdx().x

            A = CuDynamicSharedArray(Float64, (n,n))
            L = CuDynamicSharedArray(Float64, (n,n), (n^2)*sizeof(Float64))

            g = CuDynamicSharedArray(Float64, n, (2*n^2)*sizeof(Float64))
            w = CuDynamicSharedArray(Float64, n, (2*n^2 + n)*sizeof(Float64))
            p = CuDynamicSharedArray(Float64, n, (2*n^2 + 2*n)*sizeof(Float64))
            q = CuDynamicSharedArray(Float64, n, (2*n^2 + 3*n)*sizeof(Float64))
            r = CuDynamicSharedArray(Float64, n, (2*n^2 + 4*n)*sizeof(Float64))
            t = CuDynamicSharedArray(Float64, n, (2*n^2 + 5*n)*sizeof(Float64))
            z = CuDynamicSharedArray(Float64, n, (2*n^2 + 6*n)*sizeof(Float64))

            for i in 1:n
                A[i,tx] = d_in[i,tx]
                L[i,tx] = d_in[i,tx]
            end
            g[tx] = d_g[tx]
            CUDA.sync_threads()

            ExaTron.dicf(n,L)
            info, iters = ExaTron.dtrpcg(n,A,g,delta,L,tol,stol,n,w,p,q,r,t,z)
            if bx == 1
                d_out[tx] = w[tx]
                for i in 1:n
                    d_out_L[i,tx] = L[i,tx]
                end
            end
            CUDA.sync_threads()

            return
        end

        delta = 100.0
        tol = 1e-6
        stol = 1e-6
        tron_A = ExaTron.TronDenseMatrix{Array{Float64,2}}(n)
        tron_L = ExaTron.TronDenseMatrix{Array{Float64,2}}(n)
        for i=1:itermax
            L = tril(rand(n,n))
            A = L*transpose(L)
            A .= tril(A) .+ (transpose(tril(A)) .- Diagonal(A))
            g = 0.1*ones(n)
            w = zeros(n)
            p = zeros(n)
            q = zeros(n)
            r = zeros(n)
            t = zeros(n)
            z = zeros(n)
            tron_A.vals .= A
            tron_L.vals .= A
            d_in = CuArray{Float64,2}(undef, (n,n))
            d_g = CuArray{Float64}(undef, n)
            d_out_L = CuArray{Float64,2}(undef, (n,n))
            d_out = CuArray{Float64}(undef, n)
            copyto!(d_in, A)
            copyto!(d_g, g)
            CUDA.@sync @cuda threads=n blocks=nblk shmem=((2*n^2+7*n)*sizeof(Float64)) dtrpcg_test(n,delta,tol,stol,d_in,d_g,d_out_L,d_out)
            h_w = zeros(n)
            h_L = zeros(n,n)
            copyto!(h_L, d_out_L)
            copyto!(h_w, d_out)

            indr = zeros(Int, n)
            indf = zeros(n)
            list = zeros(n)
            ExaTron.dicf(n, n^2, tron_L, 5, indr, indf, list, w)
            ExaTron.dtrpcg(n, tron_A, g, delta, tron_L, tol, stol, n, w, p, q, r, t, z)

            @test norm(tril(h_L) .- tril(tron_L.vals)) <= tol
            @test norm(h_w .- w) <= tol
        end
    end

    @testset "dprsrch" begin
        function dprsrch_test(n::Int,d_x::CuDeviceArray{Float64},
                                d_xl::CuDeviceArray{Float64},
                                d_xu::CuDeviceArray{Float64},
                                d_g::CuDeviceArray{Float64},
                                d_w::CuDeviceArray{Float64},
                                d_A::CuDeviceArray{Float64},
                                d_out1::CuDeviceArray{Float64},
                                d_out2::CuDeviceArray{Float64})
            tx = CUDA.threadIdx().x
            bx = CUDA.blockIdx().x

            x = CuDynamicSharedArray(Float64, n)
            xl = CuDynamicSharedArray(Float64, n, n*sizeof(Float64))
            xu = CuDynamicSharedArray(Float64, n, (2*n)*sizeof(Float64))
            g = CuDynamicSharedArray(Float64, n, (3*n)*sizeof(Float64))
            w = CuDynamicSharedArray(Float64, n, (4*n)*sizeof(Float64))
            wa1 = CuDynamicSharedArray(Float64, n, (5*n)*sizeof(Float64))
            wa2 = CuDynamicSharedArray(Float64, n, (6*n)*sizeof(Float64))
            A = CuDynamicSharedArray(Float64, (n,n), (7*n)*sizeof(Float64))
            for i in 1:n
                A[i,tx] = d_A[i,tx]
            end
            x[tx] = d_x[tx]
            xl[tx] = d_xl[tx]
            xu[tx] = d_xu[tx]
            g[tx] = d_g[tx]
            w[tx] = d_w[tx]
            CUDA.sync_threads()

            ExaTron.dprsrch(n, x, xl, xu, A, g, w, wa1, wa2)
            if bx == 1
                d_out1[tx] = x[tx]
                d_out2[tx] = w[tx]
            end
            CUDA.sync_threads()

            return
        end

        for i=1:itermax
            L = tril(rand(n,n))
            A = ExaTron.TronDenseMatrix{Array{Float64,2}}(n)
            A.vals .= L*transpose(L)
            A.vals .= tril(A.vals) .+ (transpose(tril(A.vals)) .- Diagonal(A.vals))
            x = rand(n)
            xl = x .- abs.(rand(n))
            xu = x .+ abs.(rand(n))
            g = A.vals*x .+ rand(n)
            w = -g
            wa1 = zeros(n)
            wa2 = zeros(n)

            dx = CuArray{Float64}(undef, n)
            dl = CuArray{Float64}(undef, n)
            du = CuArray{Float64}(undef, n)
            dg = CuArray{Float64}(undef, n)
            dw = CuArray{Float64}(undef, n)
            dA = CuArray{Float64,2}(undef, (n,n))
            d_out1 = CuArray{Float64}(undef, n)
            d_out2 = CuArray{Float64}(undef, n)
            copyto!(dx, x)
            copyto!(dl, xl)
            copyto!(du, xu)
            copyto!(dg, g)
            copyto!(dw, w)
            copyto!(dA, A.vals)
            CUDA.@sync @cuda threads=n blocks=nblk shmem=((7*n+n^2)*sizeof(Float64)) dprsrch_test(n,dx,dl,du,dg,dw,dA,d_out1,d_out2)
            h_x = zeros(n)
            h_w = zeros(n)
            copyto!(h_x, d_out1)
            copyto!(h_w, d_out2)

            ExaTron.dprsrch(n,x,xl,xu,A,g,w,wa1,wa2)

            @test norm(x .- h_x) <= 1e-10
            @test norm(w .- h_w) <= 1e-10
        end
    end

    @testset "daxpy" begin
        function daxpy_test(n::Int, da, d_in::CuDeviceArray{Float64},
                            d_out::CuDeviceArray{Float64})
            tx = CUDA.threadIdx().x
            bx = CUDA.blockIdx().x

            x = CuDynamicSharedArray(Float64, n)
            y = CuDynamicSharedArray(Float64, n, n*sizeof(Float64))
            x[tx] = d_in[tx]
            y[tx] = d_in[tx + n]
            CUDA.sync_threads()

            ExaTron.daxpy(n, da, x, 1, y, 1)
            if bx == 1
                d_out[tx] = y[tx]
            end
            CUDA.sync_threads()

            return
        end

        for i=1:itermax
            da = rand(1)[1]
            h_in = rand(2*n)
            h_out = zeros(n)
            d_in = CuArray{Float64}(undef, 2*n)
            d_out = CuArray{Float64}(undef, n)
            copyto!(d_in, h_in)
            CUDA.@sync @cuda threads=n blocks=nblk shmem=((2*n)*sizeof(Float64)) daxpy_test(n,da,d_in,d_out)
            copyto!(h_out, d_out)

            @test norm(h_out .- (h_in[n+1:2*n] .+ da.*h_in[1:n])) <= 1e-12
        end
    end

    @testset "dssyax" begin
        function dssyax_test(n::Int,d_z::CuDeviceArray{Float64},
                                d_in::CuDeviceArray{Float64},
                                d_out::CuDeviceArray{Float64})
            tx = CUDA.threadIdx().x
            bx = CUDA.blockIdx().x

            z = CuDynamicSharedArray(Float64, n)
            q = CuDynamicSharedArray(Float64, n, n*sizeof(Float64))
            A = CuDynamicSharedArray(Float64, (n,n), (2*n)*sizeof(Float64))
            for i in 1:n
                A[i,tx] = d_in[i,tx]
            end
            z[tx] = d_z[tx]
            CUDA.sync_threads()

            ExaTron.dssyax(n, A, z, q)
            if bx == 1
                d_out[tx] = q[tx]
            end
            CUDA.sync_threads()

            return
        end

        for i=1:itermax
            z = rand(n)
            h_in = rand(n,n)
            h_out = zeros(n)
            d_z = CuArray{Float64}(undef, n)
            d_in = CuArray{Float64,2}(undef, (n,n))
            d_out = CuArray{Float64}(undef, n)
            copyto!(d_z, z)
            copyto!(d_in, h_in)
            CUDA.@sync @cuda threads=n blocks=nblk shmem=((2*n+n^2)*sizeof(Float64)) dssyax_test(n,d_z,d_in,d_out)
            copyto!(h_out, d_out)

            @test norm(h_out .- h_in*z) <= 1e-12
        end
    end

    @testset "dmid" begin
        function dmid_test(n::Int, dx::CuDeviceArray{Float64},
                            dl::CuDeviceArray{Float64},
                            du::CuDeviceArray{Float64},
                            d_out::CuDeviceArray{Float64})
            tx = CUDA.threadIdx().x
            bx = CUDA.blockIdx().x

            x = CuDynamicSharedArray(Float64, n)
            xl = CuDynamicSharedArray(Float64, n, n*sizeof(Float64))
            xu = CuDynamicSharedArray(Float64, n, (2*n)*sizeof(Float64))
            x[tx] = dx[tx]
            xl[tx] = dl[tx]
            xu[tx] = du[tx]
            CUDA.sync_threads()

            ExaTron.dmid(n, x, xl, xu)
            if bx == 1
                d_out[tx] = x[tx]
            end
            CUDA.sync_threads()

            return
        end

        for i=1:itermax
            x = rand(n)
            xl = x .- abs.(rand(n))
            xu = x .+ abs.(rand(n))

            # Force some components to go below or above bounds
            # so that we can test all cases.
            for j=1:n
                k = rand(1:3)
                if k == 1
                    x[j] = xl[j] - 0.1
                elseif k == 2
                    x[j] = xu[j] + 0.1
                end
            end
            x_out = zeros(n)
            dx = CuArray{Float64}(undef, n)
            dl = CuArray{Float64}(undef, n)
            du = CuArray{Float64}(undef, n)
            d_out = CuArray{Float64}(undef, n)
            copyto!(dx, x)
            copyto!(dl, xl)
            copyto!(du, xu)
            CUDA.@sync @cuda threads=n blocks=nblk shmem=((3*n)*sizeof(Float64)) dmid_test(n,dx,dl,du,d_out)
            copyto!(x_out, d_out)

            ExaTron.dmid(n, x, xl, xu)
            @test !(false in (x .== x_out))
        end
    end

    @testset "dgpstep" begin
        function dgpstep_test(n,dx::CuDeviceArray{Float64},
                                dl::CuDeviceArray{Float64},
                                du::CuDeviceArray{Float64},
                                alpha::Float64,
                                dw::CuDeviceArray{Float64},
                                d_out::CuDeviceArray{Float64})
            tx = CUDA.threadIdx().x
            bx = CUDA.blockIdx().x

            x = CuDynamicSharedArray(Float64, n)
            xl = CuDynamicSharedArray(Float64, n, n*sizeof(Float64))
            xu = CuDynamicSharedArray(Float64, n, (2*n)*sizeof(Float64))
            w = CuDynamicSharedArray(Float64, n, (3*n)*sizeof(Float64))
            s = CuDynamicSharedArray(Float64, n, (4*n)*sizeof(Float64))
            x[tx] = dx[tx]
            xl[tx] = dl[tx]
            xu[tx] = du[tx]
            w[tx] = dw[tx]
            CUDA.sync_threads()

            ExaTron.dgpstep(n, x, xl, xu, alpha, w, s)
            if bx == 1
                d_out[tx] = s[tx]
            end
            CUDA.sync_threads()

            return
        end

        for i=1:itermax
            x = rand(n)
            xl = x .- abs.(rand(n))
            xu = x .+ abs.(rand(n))
            w = rand(n)
            alpha = rand(1)[1]
            s = zeros(n)
            s_out = zeros(n)

            # Force some components to go below or above bounds
            # so that we can test all cases.
            for j=1:n
                k = rand(1:3)
                if k == 1
                    if x[j] + alpha*w[j] >= xl[j]
                        w[j] = (xl[j] - x[j]) / alpha - 0.1
                    end
                elseif k == 2
                    if x[j] + alpha*w[j] <= xu[j]
                        w[j] = (xu[j] - x[j]) / alpha + 0.1
                    end
                end
            end

            dx = CuArray{Float64}(undef, n)
            dl = CuArray{Float64}(undef, n)
            du = CuArray{Float64}(undef, n)
            dw = CuArray{Float64}(undef, n)
            d_out = CuArray{Float64}(undef, n)
            copyto!(dx, x)
            copyto!(dl, xl)
            copyto!(du, xu)
            copyto!(dw, w)
            CUDA.@sync @cuda threads=n blocks=nblk shmem=((5*n)*sizeof(Float64)) dgpstep_test(n,dx,dl,du,alpha,dw,d_out)
            copyto!(s_out, d_out)

            ExaTron.dgpstep(n, x, xl, xu, alpha, w, s)
            @test !(false in (s .== s_out))
        end
    end

    @testset "dbreakpt" begin
        function dbreakpt_test(n,dx::CuDeviceArray{Float64},
                                dl::CuDeviceArray{Float64},
                                du::CuDeviceArray{Float64},
                                dw::CuDeviceArray{Float64},
                                d_nbrpt::CuDeviceArray{Float64},
                                d_brptmin::CuDeviceArray{Float64},
                                d_brptmax::CuDeviceArray{Float64})
            tx = CUDA.threadIdx().x
            bx = CUDA.blockIdx().x

            x = CuDynamicSharedArray(Float64, n)
            xl = CuDynamicSharedArray(Float64, n, n*sizeof(Float64))
            xu = CuDynamicSharedArray(Float64, n, (2*n)*sizeof(Float64))
            w = CuDynamicSharedArray(Float64, n, (3*n)*sizeof(Float64))
            x[tx] = dx[tx]
            xl[tx] = dl[tx]
            xu[tx] = du[tx]
            w[tx] = dw[tx]
            CUDA.sync_threads()

            nbrpt, brptmin, brptmax = ExaTron.dbreakpt(n, x, xl, xu, w)
            for i in 1:n
                d_nbrpt[i,tx] = nbrpt
                d_brptmin[i,tx] = brptmin
                d_brptmax[i,tx] = brptmax
            end
            CUDA.sync_threads()

            return
        end

        for i=1:itermax
            x = rand(n)
            xl = x .- abs.(rand(n))
            xu = x .+ abs.(rand(n))
            w = 2.0*rand(n) .- 1.0     # (-1,1]
            h_nbrpt = zeros((n,n))
            h_brptmin = zeros((n,n))
            h_brptmax = zeros((n,n))

            dx = CuArray{Float64}(undef, n)
            dl = CuArray{Float64}(undef, n)
            du = CuArray{Float64}(undef, n)
            dw = CuArray{Float64}(undef, n)
            d_nbrpt = CuArray{Float64,2}(undef, (n,n))
            d_brptmin = CuArray{Float64,2}(undef, (n,n))
            d_brptmax = CuArray{Float64,2}(undef, (n,n))
            copyto!(dx, x)
            copyto!(dl, xl)
            copyto!(du, xu)
            copyto!(dw, w)
            CUDA.@sync @cuda threads=n blocks=nblk shmem=((4*n)*sizeof(Float64)) dbreakpt_test(n,dx,dl,du,dw,d_nbrpt,d_brptmin,d_brptmax)
            copyto!(h_nbrpt, d_nbrpt)
            copyto!(h_brptmin, d_brptmin)
            copyto!(h_brptmax, d_brptmax)

            nbrpt, brptmin, brptmax = ExaTron.dbreakpt(n, x, xl, xu, w)
            @test !(false in (nbrpt .== h_nbrpt))
            @test !(false in (brptmin .== h_brptmin))
            @test !(false in (brptmax .== h_brptmax))
        end
    end

    @testset "dnrm2" begin
        function dnrm2_test(n::Int, d_in::CuDeviceArray{Float64},
                            d_out::CuDeviceArray{Float64})
            tx = CUDA.threadIdx().x
            bx = CUDA.blockIdx().x

            x = CuDynamicSharedArray(Float64, n)
            x[tx] = d_in[tx]
            CUDA.sync_threads()

            v = ExaTron.dnrm2(n, x, 1)
            if bx == 1
                d_out[tx] = v
            end
            CUDA.sync_threads()

            return
        end

        @inbounds for i=1:itermax
            h_in = rand(n)
            h_out = zeros(n)
            d_in = CuArray{Float64}(undef, n)
            d_out = CuArray{Float64}(undef, n)
            copyto!(d_in, h_in)
            CUDA.@sync @cuda threads=n blocks=nblk shmem=(n*sizeof(Float64)) dnrm2_test(n,d_in,d_out)
            copyto!(h_out, d_out)
            xnorm = norm(h_in, 2)

            @test norm(xnorm .- h_out) <= 1e-10
        end
    end

    @testset "nrm2" begin
        function nrm2_test(n::Int, d_A::CuDeviceArray{Float64}, d_out::CuDeviceArray{Float64})
            tx = CUDA.threadIdx().x
            bx = CUDA.blockIdx().x

            wa = CuDynamicSharedArray(Float64, n)
            A = CuDynamicSharedArray(Float64, (n,n), n*sizeof(Float64))

            for i in 1:n
                A[i,tx] = d_A[i,tx]
            end
            CUDA.sync_threads()

            ExaTron.nrm2!(wa, A, n)
            if bx == 1
                d_out[tx] = wa[tx]
            end
            CUDA.sync_threads()

            return
        end

        @inbounds for i=1:itermax
            L = tril(rand(n,n))
            A = L*transpose(L)
            A .= tril(A) .+ (transpose(tril(A)) .- Diagonal(A))
            wa = zeros(n)
            tron_A = ExaTron.TronDenseMatrix{Array{Float64,2}}(n)
            tron_A.vals .= A
            ExaTron.nrm2!(wa, tron_A, n)

            d_A = CuArray{Float64,2}(undef, (n,n))
            d_out = CuArray{Float64}(undef, n)
            h_wa = zeros(n)
            copyto!(d_A, A)
            CUDA.@sync @cuda threads=(n,n) blocks=nblk shmem=((n^2+n)*sizeof(Float64)) nrm2_test(n,d_A,d_out)
            copyto!(h_wa, d_out)

            @test norm(wa .- h_wa) <= 1e-10
        end
    end

    @testset "dcopy" begin
        function dcopy_test(n::Int, d_in::CuDeviceArray{Float64},
                            d_out::CuDeviceArray{Float64})
            tx = CUDA.threadIdx().x
            bx = CUDA.blockIdx().x

            x = CuDynamicSharedArray(Float64, n)
            y = CuDynamicSharedArray(Float64, n, n*sizeof(Float64))

            x[tx] = d_in[tx]
            CUDA.sync_threads()

            ExaTron.dcopy(n, x, 1, y, 1)

            if bx == 1
                d_out[tx] = y[tx]
            end
            CUDA.sync_threads()

            return
        end

        @inbounds for i=1:itermax
            h_in = rand(n)
            h_out = zeros(n)
            d_in = CuArray{Float64}(undef, n)
            d_out = CuArray{Float64}(undef, n)
            copyto!(d_in, h_in)
            CUDA.@sync @cuda threads=n blocks=nblk shmem=((2*n)*sizeof(Float64)) dcopy_test(n,d_in,d_out)
            copyto!(h_out, d_out)

            @test !(false in (h_in .== h_out))
        end
    end

    @testset "ddot" begin
        function ddot_test(n::Int, d_in::CuDeviceArray{Float64},
                            d_out::CuDeviceArray{Float64})
            tx = CUDA.threadIdx().x
            bx = CUDA.blockIdx().x

            x = CuDynamicSharedArray(Float64, n)
            y = CuDynamicSharedArray(Float64, n, n*sizeof(Float64))
            x[tx] = d_in[tx]
            y[tx] = d_in[tx]
            CUDA.sync_threads()

            v = ExaTron.ddot(n, x, 1, y, 1)
            if bx == 1
                for i in 1:n
                    d_out[i,tx] = v
                end
            end
            CUDA.sync_threads()

            return
        end

        @inbounds for i=1:itermax
            h_in = rand(n)
            h_out = zeros((n,n))
            d_in = CuArray{Float64}(undef, n)
            d_out = CuArray{Float64,2}(undef, (n,n))
            copyto!(d_in, h_in)
            CUDA.@sync @cuda threads=n blocks=nblk shmem=((2*n)*sizeof(Float64)) ddot_test(n,d_in,d_out)
            copyto!(h_out, d_out)

            @test norm(dot(h_in,h_in) .- h_out, 2) <= 1e-10
        end
    end

    @testset "dscal" begin
        function dscal_test(n::Int, da::Float64,
                            d_in::CuDeviceArray{Float64},
                            d_out::CuDeviceArray{Float64})
            tx = CUDA.threadIdx().x
            bx = CUDA.blockIdx().x

            x = CuDynamicSharedArray(Float64, n)
            x[tx] = d_in[tx]
            CUDA.sync_threads()

            ExaTron.dscal(n, da, x, 1)
            if bx == 1
                d_out[tx] = x[tx]
            end
            CUDA.sync_threads()

            return
        end

        for i=1:itermax
            h_in = rand(n)
            h_out = zeros(n)
            da = rand(1)[1]
            d_in = CuArray{Float64}(undef, n)
            d_out = CuArray{Float64}(undef, n)
            copyto!(d_in, h_in)
            CUDA.@sync @cuda threads=n blocks=nblk shmem=(n*sizeof(Float64)) dscal_test(n,da,d_in,d_out)
            copyto!(h_out, d_out)

            @test norm(h_out .- (da.*h_in)) <= 1e-12
        end
    end

    @testset "dtrqsol" begin
        function dtrqsol_test(n::Int, d_x::CuDeviceArray{Float64},
                                d_p::CuDeviceArray{Float64},
                                d_out::CuDeviceArray{Float64},
                                delta::Float64)
            tx = CUDA.threadIdx().x
            ty = CUDA.threadIdx().y

            x = CuDynamicSharedArray(Float64, n)
            p = CuDynamicSharedArray(Float64, n, n*sizeof(Float64))

            x[tx] = d_x[tx]
            p[tx] = d_p[tx]
            CUDA.sync_threads()

            sigma = ExaTron.dtrqsol(n, x, p, delta)
            for i in 1:n
                d_out[i,tx] = sigma
            end
            CUDA.sync_threads()
        end

        for i=1:itermax
            x = rand(n)
            p = rand(n)
            sigma = abs(rand(1)[1])
            delta = norm(x .+ sigma.*p)

            d_x = CuArray{Float64}(undef, n)
            d_p = CuArray{Float64}(undef, n)
            d_out = CuArray{Float64,2}(undef, (n,n))
            copyto!(d_x, x)
            copyto!(d_p, p)
            CUDA.@sync @cuda threads=n blocks=nblk shmem=((2*n)*sizeof(Float64)) dtrqsol_test(n,d_x,d_p,d_out,delta)

            @test norm(sigma .- d_out) <= 1e-10
        end
    end

    @testset "dspcg" begin
        function dspcg_test(n::Int, delta::Float64, rtol::Float64,
                            cg_itermax::Int, dx::CuDeviceArray{Float64},
                            dxl::CuDeviceArray{Float64},
                            dxu::CuDeviceArray{Float64},
                            dA::CuDeviceArray{Float64},
                            dg::CuDeviceArray{Float64},
                            ds::CuDeviceArray{Float64},
                            d_out::CuDeviceArray{Float64})
            tx = CUDA.threadIdx().x
            bx = CUDA.blockIdx().x

            x = CuDynamicSharedArray(Float64, n)
            xl = CuDynamicSharedArray(Float64, n, n*sizeof(Float64))
            xu = CuDynamicSharedArray(Float64, n, (2*n)*sizeof(Float64))
            g = CuDynamicSharedArray(Float64, n, (3*n)*sizeof(Float64))
            s = CuDynamicSharedArray(Float64, n, (4*n)*sizeof(Float64))
            w = CuDynamicSharedArray(Float64, n, (5*n)*sizeof(Float64))
            wa1 = CuDynamicSharedArray(Float64, n, (6*n)*sizeof(Float64))
            wa2 = CuDynamicSharedArray(Float64, n, (7*n)*sizeof(Float64))
            wa3 = CuDynamicSharedArray(Float64, n, (8*n)*sizeof(Float64))
            wa4 = CuDynamicSharedArray(Float64, n, (9*n)*sizeof(Float64))
            wa5 = CuDynamicSharedArray(Float64, n, (10*n)*sizeof(Float64))
            gfree = CuDynamicSharedArray(Float64, n, (11*n)*sizeof(Float64))
            indfree = CuDynamicSharedArray(Int, n, (12*n)*sizeof(Float64))
            iwa = CuDynamicSharedArray(Int, 2*n, n*sizeof(Int) + (12*n)*sizeof(Float64))

            A = CuDynamicSharedArray(Float64, (n,n), (12*n)*sizeof(Float64)+(3*n)*sizeof(Int))
            B = CuDynamicSharedArray(Float64, (n,n), (12*n+n^2)*sizeof(Float64)+(3*n)*sizeof(Int))
            L = CuDynamicSharedArray(Float64, (n,n), (12*n+2*n^2)*sizeof(Float64)+(3*n)*sizeof(Int))

            @inbounds for j=1:n
                A[j,tx] = dA[j,tx]
            end
            x[tx] = dx[tx]
            xl[tx] = dxl[tx]
            xu[tx] = dxu[tx]
            g[tx] = dg[tx]
            s[tx] = ds[tx]
            CUDA.sync_threads()

            ExaTron.dspcg(n, delta, rtol, cg_itermax, x, xl, xu,
                            A, g, s, B, L, indfree, gfree, w, iwa,
                            wa1, wa2, wa3, wa4, wa5)

            if bx == 1
                d_out[tx] = x[tx]
            end
            CUDA.sync_threads()

            return
        end

        for i=1:itermax
            L = tril(rand(n,n))
            A = L*transpose(L)
            A .= tril(A) .+ (transpose(tril(A)) .- Diagonal(A))
            tron_A = ExaTron.TronDenseMatrix{Array{Float64,2}}(n)
            tron_A.vals .= A
            tron_B = ExaTron.TronDenseMatrix{Array{Float64,2}}(n)
            tron_L = ExaTron.TronDenseMatrix{Array{Float64,2}}(n)
            x = rand(n)
            xl = x .- abs.(rand(n))
            xu = x .+ abs.(rand(n))
            g = A*x .+ rand(n)
            s = rand(n)
            delta = 2.0*norm(g)
            rtol = 1e-6
            cg_itermax = n
            w = zeros(n)
            wa = zeros(5*n)
            gfree = zeros(n)
            indfree = zeros(Int, n)
            iwa = zeros(Int, 3*n)

            dx = CuArray{Float64}(undef, n)
            dxl = CuArray{Float64}(undef, n)
            dxu = CuArray{Float64}(undef, n)
            dA = CuArray{Float64,2}(undef, (n,n))
            dg = CuArray{Float64}(undef, n)
            ds = CuArray{Float64}(undef, n)
            d_out = CuArray{Float64}(undef, n)

            copyto!(dx, x)
            copyto!(dxl, xl)
            copyto!(dxu, xu)
            copyto!(dA, tron_A.vals)
            copyto!(dg, g)
            copyto!(ds, s)

            CUDA.@sync @cuda threads=n blocks=nblk shmem=((3*n)*sizeof(Int)+(12*n+3*(n^2))*sizeof(Float64)) dspcg_test(n,delta,rtol,cg_itermax,dx,dxl,dxu,dA,dg,ds,d_out)
            h_x = zeros(n)
            copyto!(h_x, d_out)

            ExaTron.dspcg(n, x, xl, xu, tron_A, g, delta, rtol, s, 5, cg_itermax,
                            tron_B, tron_L, indfree, gfree, w, wa, iwa)

            @test norm(x .- h_x) <= 1e-10
        end
    end

    @testset "dgpnorm" begin
        function dgpnorm_test(n, dx, dxl, dxu, dg, d_out)
            tx = CUDA.threadIdx().x
            bx = CUDA.blockIdx().x

            x = CuDynamicSharedArray(Float64, n)
            xl = CuDynamicSharedArray(Float64, n, n*sizeof(Float64))
            xu = CuDynamicSharedArray(Float64, n, (2*n)*sizeof(Float64))
            g = CuDynamicSharedArray(Float64, n, (3*n)*sizeof(Float64))

            x[tx] = dx[tx]
            xl[tx] = dxl[tx]
            xu[tx] = dxu[tx]
            g[tx] = dg[tx]
            CUDA.sync_threads()

            v = ExaTron.dgpnorm(n, x, xl, xu, g)
            if bx == 1
                d_out[tx] = v
            end
            CUDA.sync_threads()

            return
        end

        for i=1:itermax
            x = rand(n)
            xl = x .- abs.(rand(n))
            xu = x .+ abs.(rand(n))
            g = 2.0*rand(n) .- 1.0

            dx = CuArray{Float64}(undef, n)
            dxl = CuArray{Float64}(undef, n)
            dxu = CuArray{Float64}(undef, n)
            dg = CuArray{Float64}(undef, n)
            d_out = CuArray{Float64}(undef, n)

            copyto!(dx, x)
            copyto!(dxl, xl)
            copyto!(dxu, xu)
            copyto!(dg, g)

            gptime = @timed CUDA.@sync @cuda threads=n blocks=nblk shmem=(4*n*sizeof(Float64)) dgpnorm_test(n, dx, dxl, dxu, dg, d_out)
            h_v = zeros(n)
            copyto!(h_v, d_out)

            v = ExaTron.dgpnorm(n, x, xl, xu, g)
            @test norm(h_v .- v) <= 1e-10
        end
    end

    @testset "dtron" begin
        function dtron_test(n::Int, f::Float64, frtol::Float64, fatol::Float64, fmin::Float64,
                            cgtol::Float64, cg_itermax::Int, delta::Float64, task::Int,
                            disave::CuDeviceArray{Int}, ddsave::CuDeviceArray{Float64},
                            dx::CuDeviceArray{Float64}, dxl::CuDeviceArray{Float64},
                            dxu::CuDeviceArray{Float64}, dA::CuDeviceArray{Float64},
                            dg::CuDeviceArray{Float64}, d_out::CuDeviceArray{Float64})
            tx = CUDA.threadIdx().x
            bx = CUDA.blockIdx().x

            x = CuDynamicSharedArray(Float64, n)
            xl = CuDynamicSharedArray(Float64, n, n*sizeof(Float64))
            xu = CuDynamicSharedArray(Float64, n, (2*n)*sizeof(Float64))
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
            indfree = CuDynamicSharedArray(Int, n, (13*n)*sizeof(Float64))
            iwa = CuDynamicSharedArray(Int, 2*n, n*sizeof(Int) + (13*n)*sizeof(Float64))

            A = CuDynamicSharedArray(Float64, (n,n), (13*n)*sizeof(Float64)+(3*n)*sizeof(Int))
            B = CuDynamicSharedArray(Float64, (n,n), (13*n+n^2)*sizeof(Float64)+(3*n)*sizeof(Int))
            L = CuDynamicSharedArray(Float64, (n,n), (13*n+2*n^2)*sizeof(Float64)+(3*n)*sizeof(Int))

            @inbounds for j=1:n
                A[j,tx] = dA[j,tx]
            end
            x[tx] = dx[tx]
            xl[tx] = dxl[tx]
            xu[tx] = dxu[tx]
            g[tx] = dg[tx]
            CUDA.sync_threads()

            ExaTron.dtron(n, x, xl, xu, f, g, A, frtol, fatol, fmin, cgtol,
                            cg_itermax, delta, task, B, L, xc, s, indfree, gfree,
                            disave, ddsave, wa, iwa, wa1, wa2, wa3, wa4, wa5)

            if bx == 1
                d_out[tx] = x[tx]
            end
            CUDA.sync_threads()

            return
        end

        for i=1:itermax
            L = tril(rand(n,n))
            A = L*transpose(L)
            A .= tril(A) .+ (transpose(tril(A)) .- Diagonal(A))
            x = rand(n)
            xl = x .- abs.(rand(n))
            xu = x .+ abs.(rand(n))
            c = rand(n)
            g = A*x .+ c
            xc = zeros(n)
            s = zeros(n)
            wa = zeros(7*n)
            gfree = zeros(n)
            indfree = zeros(Int, n)
            iwa = zeros(Int, 3*n)
            isave = zeros(Int, 3)
            dsave = zeros(3)
            task = 0
            fatol = 0.0
            frtol = 1e-12
            fmin = -1e32
            cgtol = 0.1
            cg_itermax = n
            delta = 2.0*norm(g)
            f = 0.5*transpose(x)*A*x .+ transpose(x)*c

            tron_A = ExaTron.TronDenseMatrix{Array{Float64,2}}(n)
            tron_A.vals .= A
            tron_B = ExaTron.TronDenseMatrix{Array{Float64,2}}(n)
            tron_L = ExaTron.TronDenseMatrix{Array{Float64,2}}(n)

            dx = CuArray{Float64}(undef, n)
            dxl = CuArray{Float64}(undef, n)
            dxu = CuArray{Float64}(undef, n)
            dA = CuArray{Float64,2}(undef, (n,n))
            dg = CuArray{Float64}(undef, n)
            disave = CuArray{Int}(undef, n)
            ddsave = CuArray{Float64}(undef, n)
            d_out = CuArray{Float64}(undef, n)

            copyto!(dx, x)
            copyto!(dxl, xl)
            copyto!(dxu, xu)
            copyto!(dA, tron_A.vals)
            copyto!(dg, g)

            @cuda threads=n blocks=nblk shmem=((3*n)*sizeof(Int)+(13*n+3*(n^2))*sizeof(Float64)) dtron_test(n,f,frtol,fatol,fmin,cgtol,cg_itermax,delta,task,disave,ddsave,dx,dxl,dxu,dA,dg,d_out)
            h_x = zeros(n)
            copyto!(h_x, d_out)

            task_str = Vector{UInt8}(undef, 60)
            for (i,s) in enumerate("START")
                task_str[i] = UInt8(s)
            end

            ExaTron.dtron(n, x, xl, xu, f, g, tron_A, frtol, fatol, fmin, cgtol,
                            cg_itermax, delta, task_str, tron_B, tron_L, xc, s, indfree,
                            isave, dsave, wa, iwa)
            @test norm(x .- h_x) <= 1e-10
        end
    end

    @testset "driver_kernel" begin
        function eval_f(n, x, dA, dc)
            f = 0
            @inbounds for i=1:n
                @inbounds for j=1:n
                    f += x[i]*dA[i,j]*x[j]
                end
            end
            f = 0.5*f
            @inbounds for i=1:n
                f += x[i]*dc[i]
            end
            CUDA.sync_threads()
            return f
        end

        function eval_g(n, x, g, dA, dc)
            @inbounds for i=1:n
                gval = 0
                @inbounds for j=1:n
                    gval += dA[i,j]*x[j]
                end
                g[i] = gval + dc[i]
            end
            CUDA.sync_threads()
            return
        end

        function eval_h(n, scale, x, A, dA)
            tx = CUDA.threadIdx().x

            @inbounds for j=1:n
                A[j,tx] = dA[j,tx]
            end
            CUDA.sync_threads()
            return
        end

        function driver_kernel(n::Int, max_feval::Int, max_minor::Int,
                                x::CuDeviceArray{Float64}, xl::CuDeviceArray{Float64},
                                xu::CuDeviceArray{Float64}, dA::CuDeviceArray{Float64},
                                dc::CuDeviceArray{Float64})
            # We start with a shared memory allocation.
            # The first 3*n*sizeof(Float64) bytes are used for x, xl, and xu.

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
            dsave = CuDynamicSharedArray(Float64, n, (13*n)*sizeof(Float64))
            indfree = CuDynamicSharedArray(Int, n, (14*n)*sizeof(Float64))
            iwa = CuDynamicSharedArray(Int, 2*n, n*sizeof(Int) + (14*n)*sizeof(Float64))
            isave = CuDynamicSharedArray(Int, n, (3*n)*sizeof(Int) + (14*n)*sizeof(Float64))

            A = CuDynamicSharedArray(Float64, (n,n), (14*n)*sizeof(Float64)+(4*n)*sizeof(Int))
            B = CuDynamicSharedArray(Float64, (n,n), (14*n+n^2)*sizeof(Float64)+(4*n)*sizeof(Int))
            L = CuDynamicSharedArray(Float64, (n,n), (14*n+2*n^2)*sizeof(Float64)+(4*n)*sizeof(Int))

            task = 0
            status = 0

            delta = 0.0
            fatol = 0.0
            frtol = 1e-12
            fmin = -1e32
            gtol = 1e-6
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
                    f = eval_f(n, x, dA, dc)
                    nfev += 1
                    if nfev >= max_feval
                        search = false
                    end
                end

                # [2] G or H: Evaluate gradient and Hessian.

                if task == 0 || task == 2
                    eval_g(n, x, g, dA, dc)
                    eval_h(n, 1.0, x, A, dA)
                    ngev += 1
                    nhev += 1
                    minor_iter += 1
                end

                # Initialize the trust region bound.

                if task == 0
                    gnorm0 = ExaTron.dnrm2(n, g, 1)
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

                if task == 4
                    search = false
                end
            end

            return status, minor_iter
        end

        function driver_kernel_test(n, max_feval, max_minor,
                                    dx, dxl, dxu, dA, dc, d_out)
            tx = CUDA.threadIdx().x
            bx = CUDA.blockIdx().x

            x = CuDynamicSharedArray(Float64, n)
            xl = CuDynamicSharedArray(Float64, n, n*sizeof(Float64))
            xu = CuDynamicSharedArray(Float64, n, (2*n)*sizeof(Float64))

            x[tx] = dx[tx]
            xl[tx] = dxl[tx]
            xu[tx] = dxu[tx]
            CUDA.sync_threads()

            status, minor_iter = driver_kernel(n, max_feval, max_minor, x, xl, xu, dA, dc)

            if bx == 1
                d_out[tx] = x[tx]
            end
            CUDA.sync_threads()
            return
        end

        max_feval = 500
        max_minor = 100

        tron_A = ExaTron.TronDenseMatrix{Array{Float64,2}}(n)
        tron_B = ExaTron.TronDenseMatrix{Array{Float64,2}}(n)
        tron_L = ExaTron.TronDenseMatrix{Array{Float64,2}}(n)

        dx = CuArray{Float64}(undef, n)
        dxl = CuArray{Float64}(undef, n)
        dxu = CuArray{Float64}(undef, n)
        dA = CuArray{Float64,2}(undef, (n,n))
        dc = CuArray{Float64}(undef, n)
        d_out = CuArray{Float64}(undef, n)

        for i=1:itermax
            L = tril(rand(n,n))
            A = L*transpose(L)
            A .= tril(A) .+ (transpose(tril(A)) .- Diagonal(A))
            x = rand(n)
            xl = x .- abs.(rand(n))
            xu = x .+ abs.(rand(n))
            c = rand(n)

            tron_A.vals .= A

            copyto!(dx, x)
            copyto!(dxl, xl)
            copyto!(dxu, xu)
            copyto!(dA, tron_A.vals)
            copyto!(dc, c)

            function eval_f_cb(x)
                f = 0.5*(transpose(x)*A*x) + transpose(c)*x
                return f
            end

            function eval_g_cb(x, g)
                g .= A*x .+ c
            end

            function eval_h_cb(x, mode, rows, cols, scale, lambda, values)
                nz = 1
                if mode == :Structure
                    @inbounds for j=1:n
                        @inbounds for i=j:n
                            rows[nz] = i
                            cols[nz] = j
                            nz += 1
                        end
                    end
                else
                    @inbounds for j=1:n
                        @inbounds for i=j:n
                            values[nz] = A[i,j]
                            nz += 1
                        end
                    end
                end
            end

            nele_hess = div(n*(n+1), 2)
            tron = ExaTron.createProblem(n, xl, xu, nele_hess, eval_f_cb, eval_g_cb, eval_h_cb; :matrix_type=>:Dense, :max_minor=>max_minor)
            copyto!(tron.x, x)
            status = ExaTron.solveProblem(tron)

            CUDA.@sync @cuda threads=n blocks=nblk shmem=((4*n)*sizeof(Int)+(14*n+3*(n^2))*sizeof(Float64)) driver_kernel_test(n,max_feval,max_minor,dx,dxl,dxu,dA,dc,d_out)
            h_x = zeros(n)
            copyto!(h_x, d_out)

            @test norm(h_x .- tron.x) <= 1e-10
        end
    end

end
