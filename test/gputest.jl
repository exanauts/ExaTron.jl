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

@testset "GPU" begin
    itermax = 10
    n = 8
    nblk = 5120

    @testset "dicf" begin
        T = Float64
        function dicf_test(n::Int, d_in::CuDeviceArray{T},
                           d_out::CuDeviceArray{T}) where T
            tx = threadIdx().x
            ty = threadIdx().y

            L = @cuDynamicSharedMem(T, (n,n))
            L[tx,ty] = d_in[tx,ty]
            CUDA.sync_threads()

            # Test Cholesky factorization.
            ExaTron.dicf(n,L)
            d_out[tx,ty] = L[tx,ty]
            CUDA.sync_threads()
            return
        end

        for i=1:itermax
            L = tril(rand(T, n,n))
            A = L*transpose(L)
            A .= tril(A) .+ (transpose(tril(A)) .- Diagonal(A))
            tron_A = ExaTron.TronDenseMatrix{Array{T,2}}(n)
            tron_A.vals .= A

            d_in = CuArray{T,2}(undef, (n,n))
            d_out = CuArray{T,2}(undef, (n,n))
            copyto!(d_in, tron_A.vals)
            CUDA.@sync @cuda threads=(n,n) blocks=nblk shmem=(n^2*sizeof(T)) dicf_test(n,d_in,d_out)
            h_L = zeros(T, n,n)
            copyto!(h_L, d_out)

            tron_L = ExaTron.TronDenseMatrix{Array{T,2}}(n)
            tron_L.vals .= tron_A.vals
            indr = zeros(Int, n)
            indf = zeros(T, n)
            list = zeros(T, n)
            w = zeros(T, n)
            ExaTron.dicf(n, n^2, tron_L, 5, indr, indf, list, w)

            tol = eps(T)^.6
            @test norm(tron_A.vals .- tril(h_L)*transpose(tril(h_L))) <= tol
            @test norm(tril(h_L) .- transpose(triu(h_L))) <= tol
            @test norm(tril(tron_L.vals) .- tril(h_L)) <= tol
        end
    end

    @testset "dicfs" begin
        function dicfs_test(n::Int, alpha::T,
                            dA::CuDeviceArray{T},
                            d_out::CuDeviceArray{T}) where T
            tx = threadIdx().x
            ty = threadIdx().y

            wa1 = @cuDynamicSharedMem(T, n)
            wa2 = @cuDynamicSharedMem(T, n, n*sizeof(T))
            A = @cuDynamicSharedMem(T, (n,n), (2*n)*sizeof(T))
            L = @cuDynamicSharedMem(T, (n,n), (2*n+n^2)*sizeof(T))

            A[tx,ty] = dA[tx,ty]
            CUDA.sync_threads()

            ExaTron.dicfs(n, alpha, A, L, wa1, wa2)
            d_out[tx,ty] = L[tx,ty]
            CUDA.sync_threads()

            return
        end

        for i=1:itermax
            T = Float64
            L = tril(rand(T, n,n))
            A = L*transpose(L)
            A .= tril(A) .+ (transpose(tril(A)) .- Diagonal(A))
            tron_A = ExaTron.TronDenseMatrix{Array{T,2}}(n)
            tron_L = ExaTron.TronDenseMatrix{Array{T,2}}(n)
            tron_A.vals .= A

            dA = CuArray{T,2}(undef, (n,n))
            d_out = CuArray{T,2}(undef, (n,n))
            alpha = T(1.0)
            copyto!(dA, tron_A.vals)
            CUDA.@sync @cuda threads=(n,n) blocks=nblk shmem=((2*n+2*n^2)*sizeof(T)) dicfs_test(n,alpha,dA,d_out)
            h_L = zeros(T, n,n)
            copyto!(h_L, d_out)
            iwa = zeros(Int, 3*n)
            wa1 = zeros(T, n)
            wa2 = zeros(T, n)
            ExaTron.dicfs(n, n^2, tron_A, tron_L, 5, alpha, iwa, wa1, wa2)

            tol = eps(T)^.6
            @test norm(tril(h_L) .- transpose(triu(h_L))) <= tol
            @test norm(tril(tron_L.vals) .- tril(h_L)) <= tol

            # Make it negative definite.
            for j=1:n
                tron_A.vals[j,j] = -tron_A.vals[j,j]
            end
            copyto!(dA, tron_A.vals)
            CUDA.@sync @cuda threads=(n,n) blocks=nblk shmem=((2*n+2*n^2)*sizeof(T)) dicfs_test(n,alpha,dA,d_out)
            copyto!(h_L, d_out)
            ExaTron.dicfs(n, n^2, tron_A, tron_L, 5, alpha, iwa, wa1, wa2)

            @test norm(tril(h_L) .- transpose(triu(h_L))) <= tol
            @test norm(tril(tron_L.vals) .- tril(h_L)) <= tol
        end
    end

    @testset "dcauchy" begin
        function dcauchy_test(n::Int,dx::CuDeviceArray{T},
                              dl::CuDeviceArray{T},
                              du::CuDeviceArray{T},
                              dA::CuDeviceArray{T},
                              dg::CuDeviceArray{T},
                              delta::T,
                              alpha::T,
                              d_out1::CuDeviceArray{T},
                              d_out2::CuDeviceArray{T}) where T
            tx = threadIdx().x
            ty = threadIdx().y

            x = @cuDynamicSharedMem(T, n)
            xl = @cuDynamicSharedMem(T, n, n*sizeof(T))
            xu = @cuDynamicSharedMem(T, n, (2*n)*sizeof(T))
            g =  @cuDynamicSharedMem(T, n, (3*n)*sizeof(T))
            s =  @cuDynamicSharedMem(T, n, (4*n)*sizeof(T))
            wa = @cuDynamicSharedMem(T, n, (5*n)*sizeof(T))
            A =  @cuDynamicSharedMem(T, (n,n), (6*n)*sizeof(T))

            A[tx,ty] = dA[tx,ty]
            if ty == 1
                x[tx] = dx[tx]
                xl[tx] = dl[tx]
                xu[tx] = du[tx]
                g[tx] = dg[tx]
            end

            alpha = ExaTron.dcauchy(n,x,xl,xu,A,g,delta,alpha,s,wa)
            if ty == 1
                d_out1[tx] = s[tx]
                d_out2[tx] = alpha
            end
            CUDA.sync_threads()

            return
        end

        T = Float64
        for i=1:itermax
            L = tril(rand(T, n,n))
            A = ExaTron.TronDenseMatrix{Array{T,2}}(n)
            A.vals .= L*transpose(L)
            A.vals .= tril(A.vals) .+ (transpose(tril(A.vals)) .- Diagonal(A.vals))
            x = rand(T, n)
            xl = x .- abs.(rand(T, n))
            xu = x .+ abs.(rand(T, n))
            g = A.vals*x .+ rand(T, n)
            s = zeros(T, n)
            wa = zeros(T, n)
            alpha = T(1.0)
            delta = T(2.0*norm(g))

            dx = CuArray{T}(undef, n)
            dl = CuArray{T}(undef, n)
            du = CuArray{T}(undef, n)
            dg = CuArray{T}(undef, n)
            dA = CuArray{T,2}(undef, (n,n))
            d_out1 = CuArray{T}(undef, n)
            d_out2 = CuArray{T}(undef, n)
            copyto!(dx, x)
            copyto!(dl, xl)
            copyto!(du, xu)
            copyto!(dg, g)
            copyto!(dA, A.vals)
            CUDA.@sync @cuda threads=(n,n) blocks=nblk shmem=((6*n+n^2)*sizeof(T)) dcauchy_test(n,dx,dl,du,dA,dg,delta,alpha,d_out1,d_out2)
            h_s = zeros(T, n)
            h_alpha = zeros(T, n)
            copyto!(h_s, d_out1)
            copyto!(h_alpha, d_out2)

            alpha = ExaTron.dcauchy(n, x, xl, xu, A, g, delta, alpha, s, wa)

            tol = eps(T)^.6
            @test norm(s .- h_s) <= tol
            @test norm(alpha .- h_alpha) <= tol
        end
    end

    @testset "dtrpcg" begin
        function dtrpcg_test(n::Int, delta::T, tol::T,
                             stol::T, d_in::CuDeviceArray{T},
                             d_g::CuDeviceArray{T},
                             d_out_L::CuDeviceArray{T},
                             d_out::CuDeviceArray{T}) where T
            tx = threadIdx().x
            ty = threadIdx().y

            A = @cuDynamicSharedMem(T, (n,n))
            L = @cuDynamicSharedMem(T, (n,n), (n^2)*sizeof(T))

            g = @cuDynamicSharedMem(T, n, (2*n^2)*sizeof(T))
            w = @cuDynamicSharedMem(T, n, (2*n^2 + n)*sizeof(T))
            p = @cuDynamicSharedMem(T, n, (2*n^2 + 2*n)*sizeof(T))
            q = @cuDynamicSharedMem(T, n, (2*n^2 + 3*n)*sizeof(T))
            r = @cuDynamicSharedMem(T, n, (2*n^2 + 4*n)*sizeof(T))
            t = @cuDynamicSharedMem(T, n, (2*n^2 + 5*n)*sizeof(T))
            z = @cuDynamicSharedMem(T, n, (2*n^2 + 6*n)*sizeof(T))

            A[tx,ty] = d_in[tx,ty]
            L[tx,ty] = d_in[tx,ty]
            if ty == 1
                g[tx] = d_g[tx]
            end
            CUDA.sync_threads()

            ExaTron.dicf(n,L)
            info, iters = ExaTron.dtrpcg(n,A,g,delta,L,tol,stol,n,w,p,q,r,t,z)
            if ty == 1
                d_out[tx] = w[tx]
            end
            d_out_L[tx,ty] = L[tx,ty]
            CUDA.sync_threads()

            return
        end

        T = Float64
        delta = T(100.0)
        tol = T(1e-6)
        stol = T(1e-6)
        tron_A = ExaTron.TronDenseMatrix{Array{T,2}}(n)
        tron_L = ExaTron.TronDenseMatrix{Array{T,2}}(n)
        for i=1:itermax
            L = tril(rand(T,n,n))
            A = L*transpose(L)
            A .= tril(A) .+ (transpose(tril(A)) .- Diagonal(A))
            g = T(0.1)*ones(T, n)
            w = zeros(T,n)
            p = zeros(T,n)
            q = zeros(T,n)
            r = zeros(T,n)
            t = zeros(T,n)
            z = zeros(T,n)
            tron_A.vals .= A
            tron_L.vals .= A
            d_in = CuArray{T,2}(undef, (n,n))
            d_g = CuArray{T}(undef, n)
            d_out_L = CuArray{T,2}(undef, (n,n))
            d_out = CuArray{T}(undef, n)
            copyto!(d_in, A)
            copyto!(d_g, g)
            CUDA.@sync @cuda threads=(n,n) blocks=nblk shmem=((2*n^2+7*n)*sizeof(T)) dtrpcg_test(n,delta,tol,stol,d_in,d_g,d_out_L,d_out)
            h_w = zeros(T,n)
            h_L = zeros(T,n,n)
            copyto!(h_L, d_out_L)
            copyto!(h_w, d_out)

            indr = zeros(Int, n)
            indf = zeros(T,n)
            list = zeros(T,n)
            ExaTron.dicf(n, n^2, tron_L, 5, indr, indf, list, w)
            ExaTron.dtrpcg(n, tron_A, g, delta, tron_L, tol, stol, n, w, p, q, r, t, z)

            @test norm(tril(h_L) .- tril(tron_L.vals)) <= tol
            @test norm(h_w .- w) <= tol
        end
    end

    @testset "dprsrch" begin
        function dprsrch_test(n::Int,d_x::CuDeviceArray{T},
                              d_xl::CuDeviceArray{T},
                              d_xu::CuDeviceArray{T},
                              d_g::CuDeviceArray{T},
                              d_w::CuDeviceArray{T},
                              d_A::CuDeviceArray{T},
                              d_out1::CuDeviceArray{T},
                              d_out2::CuDeviceArray{T}) where T
            tx = threadIdx().x
            ty = threadIdx().y

            x = @cuDynamicSharedMem(T, n)
            xl = @cuDynamicSharedMem(T, n, n*sizeof(T))
            xu = @cuDynamicSharedMem(T, n, (2*n)*sizeof(T))
            g = @cuDynamicSharedMem(T, n, (3*n)*sizeof(T))
            w = @cuDynamicSharedMem(T, n, (4*n)*sizeof(T))
            wa1 = @cuDynamicSharedMem(T, n, (5*n)*sizeof(T))
            wa2 = @cuDynamicSharedMem(T, n, (6*n)*sizeof(T))
            A = @cuDynamicSharedMem(T, (n,n), (7*n)*sizeof(T))

            A[tx,ty] = d_A[tx,ty]
            if ty == 1
                x[tx] = d_x[tx]
                xl[tx] = d_xl[tx]
                xu[tx] = d_xu[tx]
                g[tx] = d_g[tx]
                w[tx] = d_w[tx]
            end
            CUDA.sync_threads()

            ExaTron.dprsrch(n, x, xl, xu, A, g, w, wa1, wa2)
            if ty == 1
                d_out1[tx] = x[tx]
                d_out2[tx] = w[tx]
            end
            CUDA.sync_threads()

            return
        end

        T = Float64
        for i=1:itermax
            L = tril(rand(T, n,n))
            A = ExaTron.TronDenseMatrix{Array{T,2}}(n)
            A.vals .= L*transpose(L)
            A.vals .= tril(A.vals) .+ (transpose(tril(A.vals)) .- Diagonal(A.vals))
            x = rand(T, n)
            xl = x .- abs.(rand(T, n))
            xu = x .+ abs.(rand(T, n))
            g = A.vals*x .+ rand(T, n)
            w = -g
            wa1 = zeros(T, n)
            wa2 = zeros(T, n)

            dx = CuArray{T}(undef, n)
            dl = CuArray{T}(undef, n)
            du = CuArray{T}(undef, n)
            dg = CuArray{T}(undef, n)
            dw = CuArray{T}(undef, n)
            dA = CuArray{T,2}(undef, (n,n))
            d_out1 = CuArray{T}(undef, n)
            d_out2 = CuArray{T}(undef, n)
            copyto!(dx, x)
            copyto!(dl, xl)
            copyto!(du, xu)
            copyto!(dg, g)
            copyto!(dw, w)
            copyto!(dA, A.vals)
            CUDA.@sync @cuda threads=(n,n) blocks=nblk shmem=((7*n+n^2)*sizeof(T)) dprsrch_test(n,dx,dl,du,dg,dw,dA,d_out1,d_out2)
            h_x = zeros(T, n)
            h_w = zeros(T, n)
            copyto!(h_x, d_out1)
            copyto!(h_w, d_out2)

            ExaTron.dprsrch(n,x,xl,xu,A,g,w,wa1,wa2)

            tol = eps(T)^.6
            @test norm(x .- h_x) <= tol
            @test norm(w .- h_w) <= tol
        end
    end

    @testset "daxpy" begin
        function daxpy_test(n::Int, da, d_in::CuDeviceArray{T},
                            d_out::CuDeviceArray{T}) where T
            tx = threadIdx().x
            ty = threadIdx().y

            x = @cuDynamicSharedMem(T, n)
            y = @cuDynamicSharedMem(T, n, n*sizeof(T))
            if ty == 1
                x[tx] = d_in[tx]
                y[tx] = d_in[tx + n]
            end
            CUDA.sync_threads()

            ExaTron.daxpy(n, da, x, 1, y, 1)
            if ty == 1
                d_out[tx] = y[tx]
            end
            CUDA.sync_threads()

            return
        end

        T = Float64
        for i=1:itermax
            da = rand(T, 1)[1]
            h_in = rand(T, 2*n)
            h_out = zeros(T, n)
            d_in = CuArray{T}(undef, 2*n)
            d_out = CuArray{T}(undef, n)
            copyto!(d_in, h_in)
            CUDA.@sync @cuda threads=(n,n) blocks=nblk shmem=((2*n)*sizeof(T)) daxpy_test(n,da,d_in,d_out)
            copyto!(h_out, d_out)

            @test norm(h_out .- (h_in[n+1:2*n] .+ da.*h_in[1:n])) <= 1e-12
        end
    end

    @testset "dssyax" begin
        function dssyax_test(n::Int,d_z::CuDeviceArray{T},
                             d_in::CuDeviceArray{T},
                             d_out::CuDeviceArray{T}) where T
            tx = threadIdx().x
            ty = threadIdx().y

            z = @cuDynamicSharedMem(T, n)
            q = @cuDynamicSharedMem(T, n, n*sizeof(T))
            A = @cuDynamicSharedMem(T, (n,n), (2*n)*sizeof(T))
            A[tx,ty] = d_in[tx,ty]
            if ty == 1
                z[tx] = d_z[tx]
            end
            CUDA.sync_threads()

            ExaTron.dssyax(n, A, z, q)
            if ty == 1
                d_out[tx] = q[tx]
            end
            CUDA.sync_threads()

            return
        end

        T = Float64
        for i=1:itermax
            z = rand(T, n)
            h_in = rand(T, n,n)
            h_out = zeros(T, n)
            d_z = CuArray{T}(undef, n)
            d_in = CuArray{T,2}(undef, (n,n))
            d_out = CuArray{T}(undef, n)
            copyto!(d_z, z)
            copyto!(d_in, h_in)
            CUDA.@sync @cuda threads=(n,n) blocks=nblk shmem=((2*n+n^2)*sizeof(T)) dssyax_test(n,d_z,d_in,d_out)
            copyto!(h_out, d_out)

            @test norm(h_out .- h_in*z) <= 1e-12
        end
    end

    @testset "dmid" begin
        function dmid_test(n::Int, dx::CuDeviceArray{T},
                           dl::CuDeviceArray{T},
                           du::CuDeviceArray{T},
                           d_out::CuDeviceArray{T}) where T
            tx = threadIdx().x
            ty = threadIdx().y

            x = @cuDynamicSharedMem(T, n)
            xl = @cuDynamicSharedMem(T, n, n*sizeof(T))
            xu = @cuDynamicSharedMem(T, n, (2*n)*sizeof(T))
            if ty == 1
                x[tx] = dx[tx]
                xl[tx] = dl[tx]
                xu[tx] = du[tx]
            end
            CUDA.sync_threads()

            ExaTron.dmid(n, x, xl, xu)
            if ty == 1
                d_out[tx] = x[tx]
            end
            CUDA.sync_threads()

            return
        end

        T = Float64
        for i=1:itermax
            x = rand(T, n)
            xl = x .- abs.(rand(T, n))
            xu = x .+ abs.(rand(T, n))

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
            x_out = zeros(T, n)
            dx = CuArray{T}(undef, n)
            dl = CuArray{T}(undef, n)
            du = CuArray{T}(undef, n)
            d_out = CuArray{T}(undef, n)
            copyto!(dx, x)
            copyto!(dl, xl)
            copyto!(du, xu)
            CUDA.@sync @cuda threads=(n,n) blocks=nblk shmem=((3*n)*sizeof(T)) dmid_test(n,dx,dl,du,d_out)
            copyto!(x_out, d_out)

            ExaTron.dmid(n, x, xl, xu)
            @test !(false in (x .== x_out))
        end
    end

    @testset "dgpstep" begin
        function dgpstep_test(n,dx::CuDeviceArray{T},
                              dl::CuDeviceArray{T},
                              du::CuDeviceArray{T},
                              alpha::T,
                              dw::CuDeviceArray{T},
                              d_out::CuDeviceArray{T}) where T
            tx = threadIdx().x
            ty = threadIdx().y

            x = @cuDynamicSharedMem(T, n)
            xl = @cuDynamicSharedMem(T, n, n*sizeof(T))
            xu = @cuDynamicSharedMem(T, n, (2*n)*sizeof(T))
            w = @cuDynamicSharedMem(T, n, (3*n)*sizeof(T))
            s = @cuDynamicSharedMem(T, n, (4*n)*sizeof(T))
            if ty == 1
                x[tx] = dx[tx]
                xl[tx] = dl[tx]
                xu[tx] = du[tx]
                w[tx] = dw[tx]
            end
            CUDA.sync_threads()

            ExaTron.dgpstep(n, x, xl, xu, alpha, w, s)
            if ty == 1
                d_out[tx] = s[tx]
            end
            CUDA.sync_threads()

            return
        end

        T = Float64
        for i=1:itermax
            x = rand(T, n)
            xl = x .- abs.(rand(T, n))
            xu = x .+ abs.(rand(T, n))
            w = rand(T, n)
            alpha = rand(T, 1)[1]
            s = zeros(T, n)
            s_out = zeros(T, n)

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

            dx = CuArray{T}(undef, n)
            dl = CuArray{T}(undef, n)
            du = CuArray{T}(undef, n)
            dw = CuArray{T}(undef, n)
            d_out = CuArray{T}(undef, n)
            copyto!(dx, x)
            copyto!(dl, xl)
            copyto!(du, xu)
            copyto!(dw, w)
            CUDA.@sync @cuda threads=(n,n) blocks=nblk shmem=((5*n)*sizeof(T)) dgpstep_test(n,dx,dl,du,alpha,dw,d_out)
            copyto!(s_out, d_out)

            ExaTron.dgpstep(n, x, xl, xu, alpha, w, s)
            @test !(false in (s .== s_out))
        end
    end

    @testset "dbreakpt" begin
        function dbreakpt_test(n,dx::CuDeviceArray{T},
                               dl::CuDeviceArray{T},
                               du::CuDeviceArray{T},
                               dw::CuDeviceArray{T},
                               d_nbrpt::CuDeviceArray{T},
                               d_brptmin::CuDeviceArray{T},
                               d_brptmax::CuDeviceArray{T}) where T
            tx = threadIdx().x
            ty = threadIdx().y

            x = @cuDynamicSharedMem(T, n)
            xl = @cuDynamicSharedMem(T, n, n*sizeof(T))
            xu = @cuDynamicSharedMem(T, n, (2*n)*sizeof(T))
            w = @cuDynamicSharedMem(T, n, (3*n)*sizeof(T))
            if ty == 1
                x[tx] = dx[tx]
                xl[tx] = dl[tx]
                xu[tx] = du[tx]
                w[tx] = dw[tx]
            end
            CUDA.sync_threads()

            nbrpt, brptmin, brptmax = ExaTron.dbreakpt(n, x, xl, xu, w)
            d_nbrpt[tx,ty] = nbrpt
            d_brptmin[tx,ty] = brptmin
            d_brptmax[tx,ty] = brptmax
            CUDA.sync_threads()

            return
        end

        T = Float64
        for i=1:itermax
            x = rand(T, n)
            xl = x .- abs.(rand(T, n))
            xu = x .+ abs.(rand(T, n))
            w = 2.0*rand(T, n) .- 1.0     # (-1,1]
            h_nbrpt = zeros(T, (n,n))
            h_brptmin = zeros(T, (n,n))
            h_brptmax = zeros(T, (n,n))

            dx = CuArray{T}(undef, n)
            dl = CuArray{T}(undef, n)
            du = CuArray{T}(undef, n)
            dw = CuArray{T}(undef, n)
            d_nbrpt = CuArray{T,2}(undef, (n,n))
            d_brptmin = CuArray{T,2}(undef, (n,n))
            d_brptmax = CuArray{T,2}(undef, (n,n))
            copyto!(dx, x)
            copyto!(dl, xl)
            copyto!(du, xu)
            copyto!(dw, w)
            CUDA.@sync @cuda threads=(n,n) blocks=nblk shmem=((4*n)*sizeof(T)) dbreakpt_test(n,dx,dl,du,dw,d_nbrpt,d_brptmin,d_brptmax)
            copyto!(h_nbrpt, d_nbrpt)
            copyto!(h_brptmin, d_brptmin)
            copyto!(h_brptmax, d_brptmax)

            nbrpt, brptmin, brptmax = ExaTron.dbreakpt(n, x, xl, xu, w)
            @test !(false in (nbrpt .== h_nbrpt))
            @test !(false in (brptmin .== h_brptmin))
            @test !(false in (brptmax .== h_brptmax))
        end
    end

    # @testset "dnrm2" begin
    #     function dnrm2_test(n::Int, d_in::CuDeviceArray{Float64},
    #                         d_out::CuDeviceArray{Float64})
    #         tx = threadIdx().x
    #         ty = threadIdx().y

    #         x = @cuDynamicSharedMem(Float64, n)
    #         if ty == 1
    #             x[tx] = d_in[tx]
    #         end
    #         CUDA.sync_threads()

    #         v = ExaTron.dnrm2(n, x, 1)
    #         d_out[tx,ty] = v
    #         CUDA.sync_threads()

    #         return
    #     end

    #     @inbounds for i=1:itermax
    #         h_in = rand(n)
    #         h_out = zeros((n,n))
    #         d_in = CuArray{Float64}(undef, n)
    #         d_out = CuArray{Float64,2}(undef, (n,n))
    #         copyto!(d_in, h_in)
    #         CUDA.@sync @cuda threads=(n,n) blocks=nblk shmem=(n*sizeof(Float64)) dnrm2_test(n,d_in,d_out)
    #         copyto!(h_out, d_out)
    #         xnorm = norm(h_in, 2)

    #         @test norm(xnorm .- h_out) <= 1e-10
    #     end
    # end

    @testset "nrm2" begin
        function nrm2_test(n::Int, d_A::CuDeviceArray{T}, d_out::CuDeviceArray{T}) where T
            tx = threadIdx().x
            ty = threadIdx().y

            wa = @cuDynamicSharedMem(T, n)
            A = @cuDynamicSharedMem(T, (n,n), n*sizeof(T))
            A[tx,ty] = d_A[tx,ty]
            CUDA.sync_threads()

            ExaTron.nrm2!(wa, A, n)
            if ty == 1
                d_out[tx] = wa[tx]
            end
            CUDA.sync_threads()

            return
        end

        T = Float64
        @inbounds for i=1:itermax
            L = tril(rand(T, n,n))
            A = L*transpose(L)
            A .= tril(A) .+ (transpose(tril(A)) .- Diagonal(A))
            wa = zeros(T, n)
            tron_A = ExaTron.TronDenseMatrix{Array{T,2}}(n)
            tron_A.vals .= A
            ExaTron.nrm2!(wa, tron_A, n)

            d_A = CuArray{T,2}(undef, (n,n))
            d_out = CuArray{T}(undef, n)
            h_wa = zeros(T, n)
            copyto!(d_A, A)
            CUDA.@sync @cuda threads=(n,n) blocks=nblk shmem=((n^2+n)*sizeof(T)) nrm2_test(n,d_A,d_out)
            copyto!(h_wa, d_out)

            tol = eps(T)^.6
            @test norm(wa .- h_wa) <= tol
        end
    end

    @testset "dcopy" begin
        function dcopy_test(n::Int, d_in::CuDeviceArray{T},
                            d_out::CuDeviceArray{T}) where T
            tx = threadIdx().x
            ty = threadIdx().y

            x = @cuDynamicSharedMem(T, n)
            y = @cuDynamicSharedMem(T, n, n*sizeof(T))

            if ty == 1
                x[tx] = d_in[tx]
            end
            CUDA.sync_threads()

            ExaTron.dcopy(n, x, 1, y, 1)

            if ty == 1
                d_out[tx] = y[tx]
            end
            CUDA.sync_threads()

            return
        end

        T = Float64
        @inbounds for i=1:itermax
            h_in = rand(T, n)
            h_out = zeros(T, n)
            d_in = CuArray{T}(undef, n)
            d_out = CuArray{T}(undef, n)
            copyto!(d_in, h_in)
            CUDA.@sync @cuda threads=(n,n) blocks=nblk shmem=((2*n)*sizeof(T)) dcopy_test(n,d_in,d_out)
            copyto!(h_out, d_out)

            @test !(false in (h_in .== h_out))
        end
    end

    @testset "ddot" begin
        function ddot_test(n::Int, d_in::CuDeviceArray{T},
                           d_out::CuDeviceArray{T}) where T
            tx = threadIdx().x
            ty = threadIdx().y

            x = @cuDynamicSharedMem(T, n)
            y = @cuDynamicSharedMem(T, n, n*sizeof(T))
            if ty == 1
                x[tx] = d_in[tx]
                y[tx] = d_in[tx]
            end
            CUDA.sync_threads()

            v = ExaTron.ddot(n, x, 1, y, 1)
            d_out[tx,ty] = v
            CUDA.sync_threads()

            return
        end

        T = Float64
        @inbounds for i=1:itermax
            h_in = rand(T, n)
            h_out = zeros(T, (n,n))
            d_in = CuArray{T}(undef, n)
            d_out = CuArray{T,2}(undef, (n,n))
            copyto!(d_in, h_in)
            CUDA.@sync @cuda threads=(n,n) blocks=nblk shmem=((2*n)*sizeof(T)) ddot_test(n,d_in,d_out)
            copyto!(h_out, d_out)

            tol = eps(T)^.6
            @test norm(dot(h_in,h_in) .- h_out, 2) <= tol
        end
    end

    @testset "dscal" begin
        function dscal_test(n::Int, da::T,
                            d_in::CuDeviceArray{T},
                            d_out::CuDeviceArray{T}) where T
            tx = threadIdx().x
            ty = threadIdx().y

            x = @cuDynamicSharedMem(T, n)
            if ty == 1
                x[tx] = d_in[tx]
            end
            CUDA.sync_threads()

            ExaTron.dscal(n, da, x, 1)
            if ty == 1
                d_out[tx] = x[tx]
            end
            CUDA.sync_threads()

            return
        end

        T = Float64
        for i=1:itermax
            h_in = rand(T, n)
            h_out = zeros(T, n)
            da = rand(T, 1)[1]
            d_in = CuArray{T}(undef, n)
            d_out = CuArray{T}(undef, n)
            copyto!(d_in, h_in)
            CUDA.@sync @cuda threads=(n,n) blocks=nblk shmem=(n*sizeof(T)) dscal_test(n,da,d_in,d_out)
            copyto!(h_out, d_out)

            tol = eps(T)^(2/3)
            @test norm(h_out .- (da.*h_in)) <= tol
        end
    end

    @testset "dtrqsol" begin
        function dtrqsol_test(n::Int, d_x::CuDeviceArray{T},
                              d_p::CuDeviceArray{T},
                              d_out::CuDeviceArray{T},
                              delta::T) where T
            tx = threadIdx().x
            ty = threadIdx().y

            x = @cuDynamicSharedMem(T, n)
            p = @cuDynamicSharedMem(T, n, n*sizeof(T))

            if ty == 1
                x[tx] = d_x[tx]
                p[tx] = d_p[tx]
            end
            CUDA.sync_threads()

            sigma = ExaTron.dtrqsol(n, x, p, delta)
            d_out[tx,ty] = sigma
            CUDA.sync_threads()
        end

        T = Float64
        for i=1:itermax
            x = rand(T, n)
            p = rand(T, n)
            sigma = abs(rand(T, 1)[1])
            delta = norm(x .+ sigma.*p)

            d_x = CuArray{T}(undef, n)
            d_p = CuArray{T}(undef, n)
            d_out = CuArray{T,2}(undef, (n,n))
            copyto!(d_x, x)
            copyto!(d_p, p)
            CUDA.@sync @cuda threads=(n,n) blocks=nblk shmem=((2*n)*sizeof(T)) dtrqsol_test(n,d_x,d_p,d_out,delta)

            tol = eps(T)^(2/3)
            @test norm(sigma .- d_out) <= tol
        end
    end

    @testset "dspcg" begin
        function dspcg_test(n::Int, delta::T, rtol::T,
                            cg_itermax::Int, dx::CuDeviceArray{T},
                            dxl::CuDeviceArray{T},
                            dxu::CuDeviceArray{T},
                            dA::CuDeviceArray{T},
                            dg::CuDeviceArray{T},
                            ds::CuDeviceArray{T},
                            d_out::CuDeviceArray{T}) where T
            tx = threadIdx().x
            ty = threadIdx().y

            x = @cuDynamicSharedMem(T, n)
            xl = @cuDynamicSharedMem(T, n, n*sizeof(T))
            xu = @cuDynamicSharedMem(T, n, (2*n)*sizeof(T))
            g = @cuDynamicSharedMem(T, n, (3*n)*sizeof(T))
            s = @cuDynamicSharedMem(T, n, (4*n)*sizeof(T))
            w = @cuDynamicSharedMem(T, n, (5*n)*sizeof(T))
            wa1 = @cuDynamicSharedMem(T, n, (6*n)*sizeof(T))
            wa2 = @cuDynamicSharedMem(T, n, (7*n)*sizeof(T))
            wa3 = @cuDynamicSharedMem(T, n, (8*n)*sizeof(T))
            wa4 = @cuDynamicSharedMem(T, n, (9*n)*sizeof(T))
            wa5 = @cuDynamicSharedMem(T, n, (10*n)*sizeof(T))
            gfree = @cuDynamicSharedMem(T, n, (11*n)*sizeof(T))
            indfree = @cuDynamicSharedMem(Int, n, (12*n)*sizeof(T))
            iwa = @cuDynamicSharedMem(Int, 2*n, n*sizeof(Int) + (12*n)*sizeof(T))

            A = @cuDynamicSharedMem(T, (n,n), (12*n)*sizeof(T)+(3*n)*sizeof(Int))
            B = @cuDynamicSharedMem(T, (n,n), (12*n+n^2)*sizeof(T)+(3*n)*sizeof(Int))
            L = @cuDynamicSharedMem(T, (n,n), (12*n+2*n^2)*sizeof(T)+(3*n)*sizeof(Int))

            A[tx,ty] = dA[tx,ty]
            if ty == 1
                x[tx] = dx[tx]
                xl[tx] = dxl[tx]
                xu[tx] = dxu[tx]
                g[tx] = dg[tx]
                s[tx] = ds[tx]
            end
            CUDA.sync_threads()

            ExaTron.dspcg(n, delta, rtol, cg_itermax, x, xl, xu,
                          A, g, s, B, L, indfree, gfree, w, iwa,
                          wa1, wa2, wa3, wa4, wa5)

            if ty == 1
                d_out[tx] = x[tx]
            end
            CUDA.sync_threads()

            return
        end

        T = Float64
        for i=1:itermax
            L = tril(rand(T,n,n))
            A = L*transpose(L)
            A .= tril(A) .+ (transpose(tril(A)) .- Diagonal(A))
            tron_A = ExaTron.TronDenseMatrix{Array{T,2}}(n)
            tron_A.vals .= A
            tron_B = ExaTron.TronDenseMatrix{Array{T,2}}(n)
            tron_L = ExaTron.TronDenseMatrix{Array{T,2}}(n)
            x = rand(T,n)
            xl = x .- abs.(rand(T,n))
            xu = x .+ abs.(rand(T,n))
            g = A*x .+ rand(T,n)
            s = rand(T,n)
            delta = T(2.0*norm(g))
            rtol = T(1e-6)
            cg_itermax = n
            w = zeros(T, n)
            wa = zeros(T, 5*n)
            gfree = zeros(T, n)
            indfree = zeros(Int, n)
            iwa = zeros(Int, 3*n)

            dx = CuArray{T}(undef, n)
            dxl = CuArray{T}(undef, n)
            dxu = CuArray{T}(undef, n)
            dA = CuArray{T,2}(undef, (n,n))
            dg = CuArray{T}(undef, n)
            ds = CuArray{T}(undef, n)
            d_out = CuArray{T}(undef, n)

            copyto!(dx, x)
            copyto!(dxl, xl)
            copyto!(dxu, xu)
            copyto!(dA, tron_A.vals)
            copyto!(dg, g)
            copyto!(ds, s)

            CUDA.@sync @cuda threads=(n,n) blocks=nblk shmem=((3*n)*sizeof(Int)+(12*n+3*(n^2))*sizeof(T)) dspcg_test(n,delta,rtol,cg_itermax,dx,dxl,dxu,dA,dg,ds,d_out)
            h_x = zeros(T, n)
            copyto!(h_x, d_out)

            ExaTron.dspcg(n, x, xl, xu, tron_A, g, delta, rtol, s, 5, cg_itermax,
                          tron_B, tron_L, indfree, gfree, w, wa, iwa)

            @test norm(x .- h_x) <= 1e-10
        end
    end

    @testset "dgpnorm" begin
        function dgpnorm_test(n, dx::CuDeviceVector{T, 1}, dxl, dxu, dg, d_out) where T
            tx = threadIdx().x
            ty = threadIdx().y

            x = @cuDynamicSharedMem(T, n)
            xl = @cuDynamicSharedMem(T, n, n*sizeof(T))
            xu = @cuDynamicSharedMem(T, n, (2*n)*sizeof(T))
            g = @cuDynamicSharedMem(T, n, (3*n)*sizeof(T))

            if ty == 1
                x[tx] = dx[tx]
                xl[tx] = dxl[tx]
                xu[tx] = dxu[tx]
                g[tx] = dg[tx]
            end
            CUDA.sync_threads()

            v = ExaTron.dgpnorm(n, x, xl, xu, g)
            d_out[tx] = v
            CUDA.sync_threads()

            return
        end

        T = Float64
        for i=1:itermax
            x = rand(T, n)
            xl = x .- abs.(rand(T, n))
            xu = x .+ abs.(rand(T, n))
            g = T(2.0)*rand(T, n) .- T(1.0)

            dx = CuVector{T}(undef, n)
            dxl = CuVector{T}(undef, n)
            dxu = CuVector{T}(undef, n)
            dg = CuVector{T}(undef, n)
            d_out = CuVector{T}(undef, n)

            copyto!(dx, x)
            copyto!(dxl, xl)
            copyto!(dxu, xu)
            copyto!(dg, g)

            gptime = @timed CUDA.@sync @cuda threads=(n,n) blocks=nblk shmem=(4*n*sizeof(T)) dgpnorm_test(n, dx, dxl, dxu, dg, d_out)
            h_v = zeros(T, n)
            copyto!(h_v, d_out)

            v = ExaTron.dgpnorm(n, x, xl, xu, g)
            @test norm(h_v .- v) <= 1e-10
        end
    end

    @testset "dtron" begin
        function dtron_test(n::Int, f::T, frtol::T, fatol::T, fmin::T,
                            cgtol::T, cg_itermax::Int, delta::T, task::Int,
                            disave::CuDeviceArray{Int}, ddsave::CuDeviceArray{T},
                            dx::CuDeviceArray{T}, dxl::CuDeviceArray{T},
                            dxu::CuDeviceArray{T}, dA::CuDeviceArray{T},
                            dg::CuDeviceArray{T}, d_out::CuDeviceArray{T}) where T
            tx = threadIdx().x
            ty = threadIdx().y

            x = @cuDynamicSharedMem(T, n)
            xl = @cuDynamicSharedMem(T, n, n*sizeof(T))
            xu = @cuDynamicSharedMem(T, n, (2*n)*sizeof(T))
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
            indfree = @cuDynamicSharedMem(Int, n, (13*n)*sizeof(T))
            iwa = @cuDynamicSharedMem(Int, 2*n, n*sizeof(Int) + (13*n)*sizeof(T))

            A = @cuDynamicSharedMem(T, (n,n), (13*n)*sizeof(T)+(3*n)*sizeof(Int))
            B = @cuDynamicSharedMem(T, (n,n), (13*n+n^2)*sizeof(T)+(3*n)*sizeof(Int))
            L = @cuDynamicSharedMem(T, (n,n), (13*n+2*n^2)*sizeof(T)+(3*n)*sizeof(Int))

            A[tx,ty] = dA[tx,ty]
            if ty == 1
                x[tx] = dx[tx]
                xl[tx] = dxl[tx]
                xu[tx] = dxu[tx]
                g[tx] = dg[tx]
            end
            CUDA.sync_threads()

            ExaTron.dtron(n, x, xl, xu, f, g, A, frtol, fatol, fmin, cgtol,
                          cg_itermax, delta, task, B, L, xc, s, indfree, gfree,
                          disave, ddsave, wa, iwa, wa1, wa2, wa3, wa4, wa5)
            if ty == 1
                d_out[tx] = x[tx]
            end
            CUDA.sync_threads()

            return
        end

        T = Float64
        for i=1:itermax
            L = tril(rand(T, n,n))
            A = L*transpose(L)
            A .= tril(A) .+ (transpose(tril(A)) .- Diagonal(A))
            x = rand(T, n)
            xl = x .- abs.(rand(T, n))
            xu = x .+ abs.(rand(T, n))
            c = rand(T, n)
            g = A*x .+ c
            xc = zeros(T, n)
            s = zeros(T, n)
            wa = zeros(T, 7*n)
            gfree = zeros(T, n)
            indfree = zeros(Int, n)
            iwa = zeros(Int, 3*n)
            isave = zeros(Int, 3)
            dsave = zeros(T, 3)
            task = 0
            fatol = T(0.0)
            frtol = T(1e-12)
            fmin = -T(1e32)
            cgtol = T(0.1)
            cg_itermax = n
            delta = T(2.0*norm(g))
            f = T(0.5)*transpose(x)*A*x .+ transpose(x)*c

            tron_A = ExaTron.TronDenseMatrix{Array{T,2}}(n)
            tron_A.vals .= A
            tron_B = ExaTron.TronDenseMatrix{Array{T,2}}(n)
            tron_L = ExaTron.TronDenseMatrix{Array{T,2}}(n)

            dx = CuArray{T}(undef, n)
            dxl = CuArray{T}(undef, n)
            dxu = CuArray{T}(undef, n)
            dA = CuArray{T,2}(undef, (n,n))
            dg = CuArray{T}(undef, n)
            disave = CuArray{Int}(undef, n)
            ddsave = CuArray{T}(undef, n)
            d_out = CuArray{T}(undef, n)

            copyto!(dx, x)
            copyto!(dxl, xl)
            copyto!(dxu, xu)
            copyto!(dA, tron_A.vals)
            copyto!(dg, g)

            @cuda threads=(n,n) blocks=nblk shmem=((3*n)*sizeof(Int)+(13*n+3*(n^2))*sizeof(T)) dtron_test(n,f,frtol,fatol,fmin,cgtol,cg_itermax,delta,task,disave,ddsave,dx,dxl,dxu,dA,dg,d_out)
            h_x = zeros(T, n)
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
        function eval_f(n, x::CuDeviceVector{T, 1}, dA, dc) where T
            f = T(0)
            @inbounds for i=1:n
                @inbounds for j=1:n
                    f += x[i]*dA[i,j]*x[j]
                end
            end
            f = T(0.5)*f
            @inbounds for i=1:n
                f += x[i]*dc[i]
            end
            CUDA.sync_threads()
            return f
        end

        function eval_g(n, x::CuDeviceVector{T, 1}, g, dA, dc) where T
            @inbounds for i=1:n
                gval = T(0)
                @inbounds for j=1:n
                    gval += dA[i,j]*x[j]
                end
                g[i] = gval + dc[i]
            end
            CUDA.sync_threads()
            return
        end

        function eval_h(scale, x, A, dA)
            tx = threadIdx().x
            ty = threadIdx().y

            A[tx,ty] = dA[tx,ty]
            CUDA.sync_threads()
            return
        end

        function driver_kernel(n::Int, max_feval::Int, max_minor::Int,
                               x::CuDeviceArray{T}, xl::CuDeviceArray{T},
                               xu::CuDeviceArray{T}, dA::CuDeviceArray{T},
                               dc::CuDeviceArray{T}) where T
            # We start with a shared memory allocation.
            # The first 3*n*sizeof(Float64) bytes are used for x, xl, and xu.

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

            task = 0
            status = 0

            delta = T(0.0)
            fatol = T(0.0)
            frtol = T(1e-12)
            fmin = -T(1e32)
            gtol = T(1e-6)
            cgtol = T(0.1)
            cg_itermax = n

            f = T(0.0)
            nfev = 0
            ngev = 0
            nhev = 0
            minor_iter = 0
            search = true

            while search

                # [0|1]: Evaluate function.

                if task == 0 || task == 1
                    f = eval_f(n, x, dA, dc)::T
                    nfev += 1
                    if nfev >= max_feval
                        search = false
                    end
                end

                # [2] G or H: Evaluate gradient and Hessian.

                if task == 0 || task == 2
                    eval_g(n, x, g, dA, dc)
                    eval_h(T(1.0), x, A, dA)
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
                                    dx::CuDeviceVector{T, 1}, dxl, dxu, dA, dc, d_out) where T
            tx = threadIdx().x
            ty = threadIdx().y

            x = @cuDynamicSharedMem(T, n)
            xl = @cuDynamicSharedMem(T, n, n*sizeof(T))
            xu = @cuDynamicSharedMem(T, n, (2*n)*sizeof(T))

            if ty == 1
                x[tx] = dx[tx]
                xl[tx] = dxl[tx]
                xu[tx] = dxu[tx]
            end
            CUDA.sync_threads()

            status, minor_iter = driver_kernel(n, max_feval, max_minor, x, xl, xu, dA, dc)

            if ty == 1
                d_out[tx] = x[tx]
            end
            CUDA.sync_threads()
            return
        end

        max_feval = 500
        max_minor = 100

        T = Float64
        tron_A = ExaTron.TronDenseMatrix{Array{T,2}}(n)
        tron_B = ExaTron.TronDenseMatrix{Array{T,2}}(n)
        tron_L = ExaTron.TronDenseMatrix{Array{T,2}}(n)

        dx = CuArray{T}(undef, n)
        dxl = CuArray{T}(undef, n)
        dxu = CuArray{T}(undef, n)
        dA = CuArray{T,2}(undef, (n,n))
        dc = CuArray{T}(undef, n)
        d_out = CuArray{T}(undef, n)

        for i=1:itermax
            L = tril(rand(T, n,n))
            A = L*transpose(L)
            A .= tril(A) .+ (transpose(tril(A)) .- Diagonal(A))
            x = rand(T, n)
            xl = x .- abs.(rand(T, n))
            xu = x .+ abs.(rand(T, n))
            c = rand(T, n)

            tron_A.vals .= A

            copyto!(dx, x)
            copyto!(dxl, xl)
            copyto!(dxu, xu)
            copyto!(dA, tron_A.vals)
            copyto!(dc, c)

            function eval_f_cb(x)
                f = T(0.5)*(transpose(x)*A*x) + transpose(c)*x
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
            status = ExaTron.solveProblem(tron, T)

            CUDA.@sync @cuda threads=(n,n) blocks=nblk shmem=((4*n)*sizeof(Int)+(14*n+3*(n^2))*sizeof(T)) driver_kernel_test(n,max_feval,max_minor,dx,dxl,dxu,dA,dc,d_out)
            # @device_code_warntype interactive=true @cuda driver_kernel_test(n,max_feval,max_minor,dx,dxl,dxu,dA,dc,d_out)
            h_x = zeros(T, n)
            copyto!(h_x, d_out)

            @test norm(h_x .- tron.x) <= 1e-10
        end
    end
end
