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
  - dicf     [O]: this routine also tests dnsol and dtsol.
  - dicfs    [O]
  - dcauchy  [O]
  - dtrpcg   [O]
  - dprsrch  [O]
  - daxpy    [O]
  - dssyax   [O]: we do shuffle using blockDim().x.
  - dmid     [O]: we could use a single thread only to multiple divergences.
  - dgpstep  [O]
  - dbreakpt [O]: we use the existing ExaTron implementation as is.
  - dnrm2    [O]: we do shuffle using blockDim().x.
  - nrm2     [O]: we do shuffle using blockDim().x.
  - dcopy    [O]
  - ddot     [O]
  - dscal    [O]
  - dtrqsol  [O]: we use the existing ExaTron implementation as it.
  - dspcg    [O]: we use a single thread to avoid multiple divergences.
"""

Random.seed!(0)
@testset "GPU" begin
    itermax=10
    n = 8
    nblk = 5120

    @testset "dicf" begin
        function dicf_test(n::Int, d_in::CuDeviceArray{Float64},
                           d_out::CuDeviceArray{Float64})
            tx = threadIdx().x
            ty = threadIdx().y

            L = @cuDynamicSharedMem(Float64, (n,n))
            L[tx,ty] = d_in[tx,ty]
            CUDA.sync_threads()

            # Test Cholesky factorization.
            ExaTron.dicf(n,L)
            d_out[tx,ty] = L[tx,ty]

            return
        end

        for i=1:itermax
            L = tril(rand(n,n))
            A = L*transpose(L)
            A .= tril(A) .+ (transpose(tril(A)) .- Diagonal(A))
            tron_A = ExaTron.TronDenseMatrix{Array}(n)
            tron_A.vals .= A

            d_in = CuArray{Float64}(undef, (n,n))
            d_out = CuArray{Float64}(undef, (n,n))
            copyto!(d_in, tron_A.vals)
            @cuda threads=(n,n) blocks=nblk shmem=(n^2*sizeof(Float64)) dicf_test(n,d_in,d_out)
            h_L = zeros(n,n)
            copyto!(h_L, d_out)

            tron_L = ExaTron.TronDenseMatrix{Array}(n)
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
            tx = threadIdx().x
            ty = threadIdx().y

            wa1 = @cuDynamicSharedMem(Float64, n)
            wa2 = @cuDynamicSharedMem(Float64, n, n*sizeof(Float64))
            A = @cuDynamicSharedMem(Float64, (n,n), (2*n)*sizeof(Float64))
            L = @cuDynamicSharedMem(Float64, (n,n), (2*n+n^2)*sizeof(Float64))

            A[tx,ty] = dA[tx,ty]
            CUDA.sync_threads()

            ExaTron.dicfs(n, alpha, A, L, wa1, wa2)
            d_out[tx,ty] = L[tx,ty]
            CUDA.sync_threads()

            return
        end

        for i=1:itermax
            L = tril(rand(n,n))
            A = L*transpose(L)
            A .= tril(A) .+ (transpose(tril(A)) .- Diagonal(A))
            tron_A = ExaTron.TronDenseMatrix{Array}(n)
            tron_L = ExaTron.TronDenseMatrix{Array}(n)
            tron_A.vals .= A

            dA = CuArray{Float64}(undef, (n,n))
            d_out = CuArray{Float64}(undef, (n,n))
            alpha = 1.0
            copyto!(dA, tron_A.vals)
            @cuda threads=(n,n) blocks=nblk shmem=((2*n+2*n^2)*sizeof(Float64)) dicfs_test(n,alpha,dA,d_out)
            h_L = zeros(n,n)
            copyto!(h_L, d_out)
            iwa = zeros(Int, 3*n)
            wa1 = zeros(n)
            wa2 = zeros(n)
            ExaTron.dicfs(n, n^2, tron_A, tron_L, 5, alpha, iwa, wa1, wa2)

            @test norm(tril(h_L) .- transpose(triu(h_L))) <= 1e-10
            @test norm(tril(tron_L.vals) .- tril(h_L)) <= 1e-10

            # Make it negative definite.
            for j=1:n
                tron_A.vals[j,j] = -tron_A.vals[j,j]
            end
            copyto!(dA, tron_A.vals)
            @cuda threads=(n,n) blocks=nblk shmem=((2*n+2*n^2)*sizeof(Float64)) dicfs_test(n,alpha,dA,d_out)
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
            tx = threadIdx().x
            ty = threadIdx().y

            x = @cuDynamicSharedMem(Float64, n)
            xl = @cuDynamicSharedMem(Float64, n, n*sizeof(Float64))
            xu = @cuDynamicSharedMem(Float64, n, (2*n)*sizeof(Float64))
            g =  @cuDynamicSharedMem(Float64, n, (3*n)*sizeof(Float64))
            s =  @cuDynamicSharedMem(Float64, n, (4*n)*sizeof(Float64))
            wa = @cuDynamicSharedMem(Float64, n, (5*n)*sizeof(Float64))
            A =  @cuDynamicSharedMem(Float64, (n,n), (6*n)*sizeof(Float64))

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

        for i=1:itermax
            L = tril(rand(n,n))
            A = ExaTron.TronDenseMatrix{Array}(n)
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
            dA = CuArray{Float64}(undef, (n,n))
            d_out1 = CuArray{Float64}(undef, n)
            d_out2 = CuArray{Float64}(undef, n)
            copyto!(dx, x)
            copyto!(dl, xl)
            copyto!(du, xu)
            copyto!(dg, g)
            copyto!(dA, A.vals)
            @cuda threads=(n,n) blocks=nblk shmem=((6*n+n^2)*sizeof(Float64)) dcauchy_test(n,dx,dl,du,dA,dg,delta,alpha,d_out1,d_out2)
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
                             d_out::CuDeviceArray{Float64})
            tx = threadIdx().x
            ty = threadIdx().y

            A = @cuDynamicSharedMem(Float64, (n,n))
            L = @cuDynamicSharedMem(Float64, (n,n), (n^2)*sizeof(Float64))

            g = @cuDynamicSharedMem(Float64, n, (2*n^2)*sizeof(Float64))
            w = @cuDynamicSharedMem(Float64, n, (2*n^2 + n)*sizeof(Float64))
            p = @cuDynamicSharedMem(Float64, n, (2*n^2 + 2*n)*sizeof(Float64))
            q = @cuDynamicSharedMem(Float64, n, (2*n^2 + 3*n)*sizeof(Float64))
            r = @cuDynamicSharedMem(Float64, n, (2*n^2 + 4*n)*sizeof(Float64))
            t = @cuDynamicSharedMem(Float64, n, (2*n^2 + 5*n)*sizeof(Float64))
            z = @cuDynamicSharedMem(Float64, n, (2*n^2 + 6*n)*sizeof(Float64))

            A[tx,ty] = d_in[tx,ty]
            L[tx,ty] = d_in[tx,ty]
            if ty == 1
                g[tx] = d_g[tx]
            end
            CUDA.sync_threads()

            ExaTron.dicf(n,L)
            info, iters = ExaTron.dtrpcg(n,A,g,delta,L,tol,stol,n,w,p,q,r,t,z)
            ExaTron.dtsol(n,L,w)
            if ty == 1
                d_out[tx] = w[tx]
            end
            CUDA.sync_threads()

            return
        end

        delta = 100.0
        tol = 1e-6
        stol = 1e-6
        for i=1:itermax
            L = tril(rand(n,n))
            A = L*transpose(L)
            A .= tril(A) .+ (transpose(tril(A)) .- Diagonal(A))
            g = 0.1*ones(n)
            w = zeros(n)
            d_in = CuArray{Float64}(undef, (n,n))
            d_g = CuArray{Float64}(undef, n)
            d_out = CuArray{Float64}(undef, n)
            copyto!(d_in, A)
            copyto!(d_g, g)
            @cuda threads=(n,n) blocks=nblk shmem=((2*n^2+7*n)*sizeof(Float64)) dtrpcg_test(n,delta,tol,stol,d_in,d_g,d_out)
            copyto!(w, d_out)

            @test norm(w .- A\(-g)) <= tol
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
            tx = threadIdx().x
            ty = threadIdx().y

            x = @cuDynamicSharedMem(Float64, n)
            xl = @cuDynamicSharedMem(Float64, n, n*sizeof(Float64))
            xu = @cuDynamicSharedMem(Float64, n, (2*n)*sizeof(Float64))
            g = @cuDynamicSharedMem(Float64, n, (3*n)*sizeof(Float64))
            w = @cuDynamicSharedMem(Float64, n, (4*n)*sizeof(Float64))
            wa1 = @cuDynamicSharedMem(Float64, n, (5*n)*sizeof(Float64))
            wa2 = @cuDynamicSharedMem(Float64, n, (6*n)*sizeof(Float64))
            A = @cuDynamicSharedMem(Float64, (n,n), (7*n)*sizeof(Float64))

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

        for i=1:itermax
            L = tril(rand(n,n))
            A = ExaTron.TronDenseMatrix{Array}(n)
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
            dA = CuArray{Float64}(undef, (n,n))
            d_out1 = CuArray{Float64}(undef, n)
            d_out2 = CuArray{Float64}(undef, n)
            copyto!(dx, x)
            copyto!(dl, xl)
            copyto!(du, xu)
            copyto!(dg, g)
            copyto!(dw, w)
            copyto!(dA, A.vals)
            @cuda threads=(n,n) blocks=nblk shmem=((7*n+n^2)*sizeof(Float64)) dprsrch_test(n,dx,dl,du,dg,dw,dA,d_out1,d_out2)
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
            tx = threadIdx().x
            ty = threadIdx().y

            x = @cuDynamicSharedMem(Float64, n)
            y = @cuDynamicSharedMem(Float64, n, n*sizeof(Float64))
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

        for i=1:itermax
            da = rand(1)[1]
            h_in = rand(2*n)
            h_out = zeros(n)
            d_in = CuArray{Float64}(undef, 2*n)
            d_out = CuArray{Float64}(undef, n)
            copyto!(d_in, h_in)
            @cuda threads=(n,n) blocks=nblk shmem=((2*n)*sizeof(Float64)) daxpy_test(n,da,d_in,d_out)
            copyto!(h_out, d_out)

            @test norm(h_out .- (h_in[n+1:2*n] .+ da.*h_in[1:n])) <= 1e-12
        end
    end

    @testset "dssyax" begin
        function dssyax_test(n::Int,d_z::CuDeviceArray{Float64},
                             d_in::CuDeviceArray{Float64},
                             d_out::CuDeviceArray{Float64})
            tx = threadIdx().x
            ty = threadIdx().y

            z = @cuDynamicSharedMem(Float64, n)
            q = @cuDynamicSharedMem(Float64, n, n*sizeof(Float64))
            A = @cuDynamicSharedMem(Float64, (n,n), (2*n)*sizeof(Float64))
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

        for i=1:itermax
            z = rand(n)
            h_in = rand(n,n)
            h_out = zeros(n)
            d_z = CuArray{Float64}(undef, n)
            d_in = CuArray{Float64}(undef, (n,n))
            d_out = CuArray{Float64}(undef, n)
            copyto!(d_z, z)
            copyto!(d_in, h_in)
            @cuda threads=(n,n) blocks=nblk shmem=((2*n+n^2)*sizeof(Float64)) dssyax_test(n,d_z,d_in,d_out)
            copyto!(h_out, d_out)

            @test norm(h_out .- h_in*z) <= 1e-12
        end
    end

    @testset "dmid" begin
        function dmid_test(n::Int, dx::CuDeviceArray{Float64},
                           dl::CuDeviceArray{Float64},
                           du::CuDeviceArray{Float64},
                           d_out::CuDeviceArray{Float64})
            tx = threadIdx().x
            ty = threadIdx().y

            x = @cuDynamicSharedMem(Float64, n)
            xl = @cuDynamicSharedMem(Float64, n, n*sizeof(Float64))
            xu = @cuDynamicSharedMem(Float64, n, (2*n)*sizeof(Float64))
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
            @cuda threads=(n,n) blocks=nblk shmem=((3*n)*sizeof(Float64)) dmid_test(n,dx,dl,du,d_out)
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
            tx = threadIdx().x
            ty = threadIdx().y

            x = @cuDynamicSharedMem(Float64, n)
            xl = @cuDynamicSharedMem(Float64, n, n*sizeof(Float64))
            xu = @cuDynamicSharedMem(Float64, n, (2*n)*sizeof(Float64))
            w = @cuDynamicSharedMem(Float64, n, (3*n)*sizeof(Float64))
            s = @cuDynamicSharedMem(Float64, n, (4*n)*sizeof(Float64))
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
            @cuda threads=(n,n) blocks=nblk shmem=((5*n)*sizeof(Float64)) dgpstep_test(n,dx,dl,du,alpha,dw,d_out)
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
            tx = threadIdx().x
            ty = threadIdx().y

            x = @cuDynamicSharedMem(Float64, n)
            xl = @cuDynamicSharedMem(Float64, n, n*sizeof(Float64))
            xu = @cuDynamicSharedMem(Float64, n, (2*n)*sizeof(Float64))
            w = @cuDynamicSharedMem(Float64, n, (3*n)*sizeof(Float64))
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
            d_nbrpt = CuArray{Float64}(undef, (n,n))
            d_brptmin = CuArray{Float64}(undef, (n,n))
            d_brptmax = CuArray{Float64}(undef, (n,n))
            copyto!(dx, x)
            copyto!(dl, xl)
            copyto!(du, xu)
            copyto!(dw, w)
            @cuda threads=(n,n) blocks=nblk shmem=((4*n)*sizeof(Float64)) dbreakpt_test(n,dx,dl,du,dw,d_nbrpt,d_brptmin,d_brptmax)
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
            tx = threadIdx().x
            ty = threadIdx().y

            x = @cuDynamicSharedMem(Float64, n)
            if ty == 1
                x[tx] = d_in[tx]
            end
            CUDA.sync_threads()

            v = ExaTron.dnrm2(n, x, 1)
            d_out[tx,ty] = v
            CUDA.sync_threads()

            return
        end

        @inbounds for i=1:itermax
            h_in = rand(n)
            h_out = zeros((n,n))
            d_in = CuArray{Float64}(undef, n)
            d_out = CuArray{Float64}(undef, (n,n))
            copyto!(d_in, h_in)
            @cuda threads=(n,n) blocks=nblk shmem=(n*sizeof(Float64)) dnrm2_test(n,d_in,d_out)
            copyto!(h_out, d_out)
            xnorm = norm(h_in, 2)

            @test norm(xnorm .- h_out) <= 1e-10
        end
    end

    @testset "nrm2" begin
        function nrm2_test(n::Int, d_A::CuDeviceArray{Float64}, d_out::CuDeviceArray{Float64})
            tx = threadIdx().x
            ty = threadIdx().y

            wa = @cuDynamicSharedMem(Float64, n)
            A = @cuDynamicSharedMem(Float64, (n,n), n*sizeof(Float64))
            A[tx,ty] = d_A[tx,ty]
            CUDA.sync_threads()

            ExaTron.nrm2!(wa, A, n)
            if ty == 1
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
            tron_A = ExaTron.TronDenseMatrix{Array}(n)
            tron_A.vals .= A
            ExaTron.nrm2!(wa, tron_A, n)

            d_A = CuArray{Float64}(undef, (n,n))
            d_out = CuArray{Float64}(undef, n)
            h_wa = zeros(n)
            copyto!(d_A, A)
            @cuda threads=(n,n) blocks=nblk shmem=((n^2+n)*sizeof(Float64)) nrm2_test(n,d_A,d_out)
            copyto!(h_wa, d_out)

            @test norm(wa .- h_wa) <= 1e-10
        end
    end

    @testset "dcopy" begin
        function dcopy_test(n::Int, d_in::CuDeviceArray{Float64},
                            d_out::CuDeviceArray{Float64})
            tx = threadIdx().x
            ty = threadIdx().y

            x = @cuDynamicSharedMem(Float64, n)
            y = @cuDynamicSharedMem(Float64, n, n*sizeof(Float64))

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

        @inbounds for i=1:itermax
            h_in = rand(n)
            h_out = zeros(n)
            d_in = CuArray{Float64}(undef, n)
            d_out = CuArray{Float64}(undef, n)
            copyto!(d_in, h_in)
            @cuda threads=(n,n) blocks=nblk shmem=((2*n)*sizeof(Float64)) dcopy_test(n,d_in,d_out)
            copyto!(h_out, d_out)

            @test !(false in (h_in .== h_out))
        end
    end

    @testset "ddot" begin
        function ddot_test(n::Int, d_in::CuDeviceArray{Float64},
                           d_out::CuDeviceArray{Float64})
            tx = threadIdx().x
            ty = threadIdx().y

            x = @cuDynamicSharedMem(Float64, n)
            y = @cuDynamicSharedMem(Float64, n, n*sizeof(Float64))
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

        @inbounds for i=1:itermax
            h_in = rand(n)
            h_out = zeros((n,n))
            d_in = CuArray{Float64}(undef, n)
            d_out = CuArray{Float64}(undef, (n,n))
            copyto!(d_in, h_in)
            @cuda threads=(n,n) blocks=nblk shmem=((2*n)*sizeof(Float64)) ddot_test(n,d_in,d_out)
            copyto!(h_out, d_out)

            @test norm(dot(h_in,h_in) .- h_out, 2) <= 1e-10
        end
    end

    @testset "dscal" begin
        function dscal_test(n::Int, da::Float64,
                            d_in::CuDeviceArray{Float64},
                            d_out::CuDeviceArray{Float64})
            tx = threadIdx().x
            ty = threadIdx().y

            x = @cuDynamicSharedMem(Float64, n)
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

        for i=1:itermax
            h_in = rand(n)
            h_out = zeros(n)
            da = rand(1)[1]
            d_in = CuArray{Float64}(undef, n)
            d_out = CuArray{Float64}(undef, n)
            copyto!(d_in, h_in)
            @cuda threads=(n,n) blocks=nblk shmem=(n*sizeof(Float64)) dscal_test(n,da,d_in,d_out)
            copyto!(h_out, d_out)

            @test norm(h_out .- (da.*h_in)) <= 1e-12
        end
    end

    @testset "dtrqsol" begin
        function dtrqsol_test(n::Int, d_x::CuDeviceArray{Float64},
                              d_p::CuDeviceArray{Float64},
                              d_out::CuDeviceArray{Float64},
                              delta::Float64)
            tx = threadIdx().x
            ty = threadIdx().y

            x = @cuDynamicSharedMem(Float64, n)
            p = @cuDynamicSharedMem(Float64, n, n*sizeof(Float64))

            if ty == 1
                x[tx] = d_x[tx]
                p[tx] = d_p[tx]
            end
            CUDA.sync_threads()

            sigma = ExaTron.dtrqsol(n, x, p, delta)
            d_out[tx,ty] = sigma
            CUDA.sync_threads()
        end

        for i=1:itermax
            x = rand(n)
            p = rand(n)
            sigma = abs(rand(1)[1])
            delta = norm(x .+ sigma.*p)

            d_x = CuArray{Float64}(undef, n)
            d_p = CuArray{Float64}(undef, n)
            d_out = CuArray{Float64}(undef, (n,n))
            copyto!(d_x, x)
            copyto!(d_p, p)
            @cuda threads=(n,n) blocks=nblk shmem=((2*n)*sizeof(Float64)) dtrqsol_test(n,d_x,d_p,d_out,delta)

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
            tx = threadIdx().x
            ty = threadIdx().y

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

        for i=1:itermax
            L = tril(rand(n,n))
            A = L*transpose(L)
            A .= tril(A) .+ (transpose(tril(A)) .- Diagonal(A))
            tron_A = ExaTron.TronDenseMatrix{Array}(n)
            tron_A.vals .= A
            tron_B = ExaTron.TronDenseMatrix{Array}(n)
            tron_L = ExaTron.TronDenseMatrix{Array}(n)
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
            dA = CuArray{Float64}(undef, (n,n))
            dg = CuArray{Float64}(undef, n)
            ds = CuArray{Float64}(undef, n)
            d_out = CuArray{Float64}(undef, n)

            copyto!(dx, x)
            copyto!(dxl, xl)
            copyto!(dxu, xu)
            copyto!(dA, tron_A.vals)
            copyto!(dg, g)
            copyto!(ds, s)

            @cuda threads=(n,n) blocks=nblk shmem=((3*n)*sizeof(Int)+(12*n+3*(n^2))*sizeof(Float64)) dspcg_test(n,delta,rtol,cg_itermax,dx,dxl,dxu,dA,dg,ds,d_out)
            h_x = zeros(n)
            copyto!(h_x, d_out)

            ExaTron.dspcg(n, x, xl, xu, tron_A, g, delta, rtol, s, 5, cg_itermax,
                          tron_B, tron_L, indfree, gfree, w, wa, iwa)

            @test norm(x .- h_x) <= 1e-10
        end
    end
end
