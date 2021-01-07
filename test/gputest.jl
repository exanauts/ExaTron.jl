try
    tmp = CuArray{Float64}(undef, 10)
catch e
    throw(e)
end

# Test routines:
#   dicf (includes dnsol and dtsol)
#   dtrpcg
#   daxpy
#   dssyax
#   dmid
#   dnrm2
#   dcopy
#   ddot
#   dscal
#   dtrqsol

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

        @inbounds for i=1:itermax
            L = tril(rand(n,n))
            A = L*transpose(L)
            B = zeros(n,n)
            d_in = CuArray{Float64}(undef, (n,n))
            d_out = CuArray{Float64}(undef, (n,n))
            copyto!(d_in, A)
            @cuda threads=(n,n) blocks=nblk shmem=(n^2*sizeof(Float64)) dicf_test(n,d_in,d_out)
            copyto!(B, d_out)

            @test norm(A .- tril(B)*transpose(tril(B))) <= 1e-10
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
                d_out[tx] =x[tx]
            end
            CUDA.sync_threads()

            return
        end

        for i=1:itermax
            x = rand(n)
            xl = x .- abs.(rand(n))
            xu = x .+ abs.(rand(n))
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
end
