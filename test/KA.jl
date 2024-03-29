using AMDGPU
using CUDA
using ExaTron
using KernelAbstractions
using LinearAlgebra
using Random
using Test

const KA = KernelAbstractions

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
itermax = 10
n = 4
nblk = 4

if has_cuda_gpu()
    device = CUDABackend()
    AT = CuArray
elseif has_rocm_gpu()
    # Set for crusher login node to avoid other users
    device = AMDGPU.ROCBackend()
    AT = ROCArray
else
    device = CPU()
    error("CPU KA implementation is currently broken for nested functions")
end


@testset "dicf" begin
@kernel function dicf_test(::Val{n}, d_in,
                    d_out) where {n}
    tx = @index(Local, Linear)
    bx = @index(Group, Linear)

    L = @localmem Float64 (n,n)
    for i in 1:n
        L[tx,i] = d_in[tx,i]
    end
    @synchronize

    # Test Cholesky factorization.
    ExaTron.dicf(n,L,tx)
    if bx == 1
        for i in 1:n
            d_out[tx,i] = L[tx,i]
        end
    end
    @synchronize
end

    for i=1:itermax
        L = tril(rand(n,n))
        A = L*transpose(L)
        A .= tril(A) .+ (transpose(tril(A)) .- Diagonal(A))
        tron_A = ExaTron.TronDenseMatrix{Array{Float64,2}}(n)
        tron_A.vals .= A

        d_in = AT{Float64,2}(undef, (n,n))
        d_out = AT{Float64,2}(undef, (n,n))
        copyto!(d_in, tron_A.vals)
        dicf_test(device, n)(Val{n}(), d_in, d_out, ndrange=(n,nblk))
        KA.synchronize(device)
        h_L = d_out |> Array

        tron_L = ExaTron.TronDenseMatrix{Array{Float64,2}}(n)
        tron_L.vals .= tron_A.vals
        indr = zeros(Int, n)
        indf = zeros(n)
        list = zeros(n)
        w = zeros(n)
        ExaTron.dicf(n, n^2, tron_L, 5, indr, indf, list, w)

        @test norm(tron_A.vals .- tril(h_L)*transpose(tril(h_L))) <= 1e-10
        @test norm(tril(h_L) .- transpose(triu(h_L))) <= 1e-10
        @test norm(tril(tron_L.vals) .- tril(h_L)) <= 1e-9
    end
end

@testset "dicfs" begin
    @kernel function dicfs_test(::Val{n}, alpha::Float64,
                        dA,
                        d_out) where {n}
        tx = @index(Local, Linear)
        bx = @index(Group, Linear)

        wa1 = @localmem Float64 (n,)
        wa2 = @localmem Float64 (n,)
        A = @localmem Float64 (n,n)
        L = @localmem Float64 (n,n)

        for i in 1:n
            A[tx,i] = dA[tx,i]
        end
        @synchronize

        ExaTron.dicfs(n, alpha, A, L, wa1, wa2, tx)
        if bx <= 1
            for i in 1:n
                d_out[tx,i] = L[tx,i]
            end
        end
        @synchronize
    end

    for i=1:itermax
        L = tril(rand(n,n))
        A = L*transpose(L)
        A .= tril(A) .+ (transpose(tril(A)) .- Diagonal(A))
        tron_A = ExaTron.TronDenseMatrix{Array{Float64,2}}(n)
        tron_L = ExaTron.TronDenseMatrix{Array{Float64,2}}(n)
        tron_A.vals .= A

        dA = AT{Float64,2}(undef, (n,n))
        d_out = AT{Float64,2}(undef, (n,n))
        alpha = 1.0
        copyto!(dA, tron_A.vals)
        dicfs_test(device, n)(Val{n}(),alpha,dA,d_out,ndrange=(n, nblk))
        KA.synchronize(device)
        h_L = d_out |> Array
        iwa = zeros(Int, 3*n)
        wa1 = zeros(n)
        wa2 = zeros(n)
        ExaTron.dicfs(n, n*n, tron_A, tron_L, 5, alpha, iwa, wa1, wa2)

        @test norm(tril(h_L) .- transpose(triu(h_L))) <= 1e-10
        @test norm(tril(tron_L.vals) .- tril(h_L)) <= 1e-9

        # Make it negative definite.
        for j=1:n
            tron_A.vals[j,j] = -tron_A.vals[j,j]
        end
        copyto!(dA, tron_A.vals)
        dicfs_test(device, n)(Val{n}(),alpha,dA,d_out,ndrange=(n,nblk))
        KA.synchronize(device)
        copyto!(h_L, d_out)
        ExaTron.dicfs(n, n^2, tron_A, tron_L, 5, alpha, iwa, wa1, wa2)

        @test norm(tril(h_L) .- transpose(triu(h_L))) <= 1e-10
        @test norm(tril(tron_L.vals) .- tril(h_L)) <= 1e-10
    end
end

@testset "dcauchy" begin
    @kernel function dcauchy_test(::Val{n},dx,
                            dl,
                            du,
                            dA,
                            dg,
                            delta::Float64,
                            alpha::Float64,
                            d_out1,
                            d_out2
                            ) where {n}
        tx = @index(Local, Linear)
        bx = @index(Group, Linear)

        x = @localmem Float64 (n,)
        xl = @localmem Float64 (n,)
        xu = @localmem Float64 (n,)
        g = @localmem Float64 (n,)
        s = @localmem Float64 (n,)
        wa = @localmem Float64 (n,)
        A = @localmem Float64 (n,n)
        for i in 1:n
            A[tx,i] = dA[tx,i]
        end
        x[tx] = dx[tx]
        xl[tx] = dl[tx]
        xu[tx] = du[tx]
        g[tx] = dg[tx]

        alpha = ExaTron.dcauchy(n,x,xl,xu,A,g,delta,alpha,s,wa,tx)
        if bx == 1
            d_out1[tx] = s[tx]
            d_out2[tx] = alpha
        end
        @synchronize
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

        dx = AT{Float64}(undef, n)
        dl = AT{Float64}(undef, n)
        du = AT{Float64}(undef, n)
        dg = AT{Float64}(undef, n)
        dA = AT{Float64,2}(undef, (n,n))
        d_out1 = AT{Float64}(undef, n)
        d_out2 = AT{Float64}(undef, n)
        copyto!(dx, x)
        copyto!(dl, xl)
        copyto!(du, xu)
        copyto!(dg, g)
        copyto!(dA, A.vals)
        dcauchy_test(device, n)(Val{n}(),dx,dl,du,dA,dg,delta,alpha,d_out1,d_out2,ndrange=(n,nblk))
        KA.synchronize(device)
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
    @kernel function dtrpcg_test(::Val{n}, delta::Float64, tol::Float64,
                            stol::Float64, d_in,
                            d_g,
                            d_out_L,
                            d_out
                            ) where {n}
        tx = @index(Local, Linear)
        bx = @index(Group, Linear)

        A = @localmem Float64 (n,n)
        L = @localmem Float64 (n,n)

        g = @localmem Float64 (n,)
        w = @localmem Float64 (n,)
        p = @localmem Float64 (n,)
        q = @localmem Float64 (n,)
        r = @localmem Float64 (n,)
        t = @localmem Float64 (n,)
        z = @localmem Float64 (n,)
        for i in 1:n
            A[tx,i] = d_in[tx,i]
            L[tx,i] = d_in[tx,i]
        end
        g[tx] = d_g[tx]
        @synchronize

        ExaTron.dicf(n,L,tx)
        info, iters = ExaTron.dtrpcg(n,A,g,delta,L,tol,stol,n,w,p,q,r,t,z,tx)
        if bx == 1
            d_out[tx] = w[tx]
            for i in 1:n
                d_out_L[tx,i] = L[tx,i]
            end
        end
        @synchronize
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
        d_in = AT{Float64,2}(undef, (n,n))
        d_g = AT{Float64}(undef, n)
        d_out_L = AT{Float64,2}(undef, (n,n))
        d_out = AT{Float64}(undef, n)
        copyto!(d_in, A)
        copyto!(d_g, g)
        dtrpcg_test(device, n)(Val{n}(),delta,tol,stol,d_in,d_g,d_out_L,d_out,ndrange=(n,nblk))
        KA.synchronize(device)
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
    @kernel function dprsrch_test(::Val{n},d_x,
                            d_xl,
                            d_xu,
                            d_g,
                            d_w,
                            d_A,
                            d_out1,
                            d_out2
                            ) where {n}
        tx = @index(Local, Linear)
        bx = @index(Group, Linear)

        x = @localmem Float64 (n,)
        xl = @localmem Float64 (n,)
        xu = @localmem Float64 (n,)
        g = @localmem Float64 (n,)
        w = @localmem Float64 (n,)
        wa1 = @localmem Float64 (n,)
        wa2 = @localmem Float64 (n,)
        A = @localmem Float64 (n,n)

        for i in 1:n
            A[tx,i] = d_A[tx,i]
        end
        x[tx] = d_x[tx]
        xl[tx] = d_xl[tx]
        xu[tx] = d_xu[tx]
        g[tx] = d_g[tx]
        w[tx] = d_w[tx]
        @synchronize

        ExaTron.dprsrch(n, x, xl, xu, A, g, w, wa1, wa2, tx)
        if bx == 1
            d_out1[tx] = x[tx]
            d_out2[tx] = w[tx]
        end
        @synchronize

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

        dx = AT{Float64}(undef, n)
        dl = AT{Float64}(undef, n)
        du = AT{Float64}(undef, n)
        dg = AT{Float64}(undef, n)
        dw = AT{Float64}(undef, n)
        dA = AT{Float64,2}(undef, (n,n))
        d_out1 = AT{Float64}(undef, n)
        d_out2 = AT{Float64}(undef, n)
        copyto!(dx, x)
        copyto!(dl, xl)
        copyto!(du, xu)
        copyto!(dg, g)
        copyto!(dw, w)
        copyto!(dA, A.vals)
        dprsrch_test(device, n)(Val{n}(),dx,dl,du,dg,dw,dA,d_out1,d_out2,ndrange=(n,nblk))
        KA.synchronize(device)
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
    @kernel function daxpy_test(::Val{n}, da, d_in,
                        d_out
                        ) where {n}
        tx = @index(Local, Linear)
        bx = @index(Group, Linear)

        x = @localmem Float64 (n,)
        y = @localmem Float64 (n,)
        x[tx] = d_in[tx]
        y[tx] = d_in[tx + n]
        @synchronize

        ExaTron.daxpy(n, da, x, 1, y, 1, tx)
        if bx == 1
            d_out[tx] = y[tx]
        end
        @synchronize

    end

    for i=1:itermax
        da = rand(1)[1]
        h_in = rand(2*n)
        h_out = zeros(n)
        d_in = AT{Float64}(undef, 2*n)
        d_out = AT{Float64}(undef, n)
        copyto!(d_in, h_in)
        daxpy_test(device,n)(Val{n}(),da,d_in,d_out,ndrange=(n,nblk))
        KA.synchronize(device)
        copyto!(h_out, d_out)

        @test norm(h_out .- (h_in[n+1:2*n] .+ da.*h_in[1:n])) <= 1e-12
    end
end

@testset "dssyax" begin
    @kernel function dssyax_test(::Val{n},d_z,
                            d_in,
                            d_out
                            ) where {n}
        tx = @index(Local, Linear)
        bx = @index(Group, Linear)

        z = @localmem Float64 (n,)
        q = @localmem Float64 (n,)
        A = @localmem Float64 (n,n)

        for i in 1:n
            A[tx,i] = d_in[tx,i]
        end
        z[tx] = d_z[tx]
        @synchronize

        ExaTron.dssyax(n, A, z, q, tx)
        if bx == 1
            d_out[tx] = q[tx]
        end
        @synchronize

    end

    for i=1:itermax
        z = rand(n)
        h_in = rand(n,n)
        h_out = zeros(n)
        d_z = AT{Float64}(undef, n)
        d_in = AT{Float64,2}(undef, (n,n))
        d_out = AT{Float64}(undef, n)
        copyto!(d_z, z)
        copyto!(d_in, h_in)
        dssyax_test(device,n)(Val{n}(),d_z,d_in,d_out,ndrange=(n,nblk))
        KA.synchronize(device)
        copyto!(h_out, d_out)

        @test norm(h_out .- h_in*z) <= 1e-12
    end
end

@testset "dmid" begin
    @kernel function dmid_test(::Val{n}, dx,
                        dl,
                        du,
                        d_out
                        ) where {n}
        tx = @index(Local, Linear)
        bx = @index(Group, Linear)

        x = @localmem Float64 (n,)
        xl = @localmem Float64 (n,)
        xu = @localmem Float64 (n,)
        x[tx] = dx[tx]
        xl[tx] = dl[tx]
        xu[tx] = du[tx]
        @synchronize

        ExaTron.dmid(n, x, xl, xu, tx)
        if bx == 1
            d_out[tx] = x[tx]
        end
        @synchronize

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
        dx = AT{Float64}(undef, n)
        dl = AT{Float64}(undef, n)
        du = AT{Float64}(undef, n)
        d_out = AT{Float64}(undef, n)
        copyto!(dx, x)
        copyto!(dl, xl)
        copyto!(du, xu)
        dmid_test(device,n)(Val{n}(),dx,dl,du,d_out,ndrange=(n,nblk))
        KA.synchronize(device)
        copyto!(x_out, d_out)

        ExaTron.dmid(n, x, xl, xu)
        @test !(false in (x .== x_out))
    end
end

@testset "dgpstep" begin
    @kernel function dgpstep_test(::Val{n},dx,
                            dl,
                            du,
                            alpha::Float64,
                            dw,
                            d_out
                            ) where {n}
        tx = @index(Local, Linear)
        bx = @index(Group, Linear)

        x = @localmem Float64 (n,)
        xl = @localmem Float64 (n,)
        xu = @localmem Float64 (n,)
        w = @localmem Float64 (n,)
        s = @localmem Float64 (n,)
        x[tx] = dx[tx]
        xl[tx] = dl[tx]
        xu[tx] = du[tx]
        w[tx] = dw[tx]
        @synchronize

        ExaTron.dgpstep(n, x, xl, xu, alpha, w, s, tx)
        if bx == 1
            d_out[tx] = s[tx]
        end
        @synchronize

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

        dx = AT{Float64}(undef, n)
        dl = AT{Float64}(undef, n)
        du = AT{Float64}(undef, n)
        dw = AT{Float64}(undef, n)
        d_out = AT{Float64}(undef, n)
        copyto!(dx, x)
        copyto!(dl, xl)
        copyto!(du, xu)
        copyto!(dw, w)
        dgpstep_test(device,n)(Val{n}(),dx,dl,du,alpha,dw,d_out,ndrange=(n,nblk))
        KA.synchronize(device)
        copyto!(s_out, d_out)

        ExaTron.dgpstep(n, x, xl, xu, alpha, w, s)
        @test !(false in (s .== s_out))
    end
end

@testset "dbreakpt" begin
    @kernel function dbreakpt_test(::Val{n},dx,
                            dl,
                            du,
                            dw,
                            d_nbrpt,
                            d_brptmin,
                            d_brptmax
                            ) where {n}
        tx = @index(Local, Linear)
        bx = @index(Group, Linear)

        x = @localmem Float64 (n,)
        xl = @localmem Float64 (n,)
        xu = @localmem Float64 (n,)
        w = @localmem Float64 (n,)
        x[tx] = dx[tx]
        xl[tx] = dl[tx]
        xu[tx] = du[tx]
        w[tx] = dw[tx]
        @synchronize

        nbrpt, brptmin, brptmax = ExaTron.dbreakpt(n, x, xl, xu, w, tx)
        if bx == 1
            for i in 1:n
                d_nbrpt[tx,i] = nbrpt
                d_brptmin[tx,i] = brptmin
                d_brptmax[tx,i] = brptmax
            end
        end
        @synchronize

    end

    for i=1:itermax
        x = rand(n)
        xl = x .- abs.(rand(n))
        xu = x .+ abs.(rand(n))
        w = 2.0*rand(n) .- 1.0     # (-1,1]
        h_nbrpt = zeros((n,n))
        h_brptmin = zeros((n,n))
        h_brptmax = zeros((n,n))

        dx = AT{Float64}(undef, n)
        dl = AT{Float64}(undef, n)
        du = AT{Float64}(undef, n)
        dw = AT{Float64}(undef, n)
        d_nbrpt = AT{Float64,2}(undef, (n,n))
        d_brptmin = AT{Float64,2}(undef, (n,n))
        d_brptmax = AT{Float64,2}(undef, (n,n))
        copyto!(dx, x)
        copyto!(dl, xl)
        copyto!(du, xu)
        copyto!(dw, w)
        dbreakpt_test(device,n)(Val{n}(),dx,dl,du,dw,d_nbrpt,d_brptmin,d_brptmax,ndrange=(n,nblk))
        KA.synchronize(device)
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
    @kernel function dnrm2_test(::Val{n}, d_in,
                        d_out
                        ) where {n}
        tx = @index(Local, Linear)
        bx = @index(Group, Linear)

        x = @localmem Float64 (n,)
        x[tx] = d_in[tx]
        @synchronize

        v = ExaTron.dnrm2(n, x, 1, tx)
        if bx == 1
            for i in 1:n
                d_out[tx,i] = v
            end
        end
        @synchronize

    end

    @inbounds for i=1:itermax
        h_in = rand(n)
        h_out = zeros((n,n))
        d_in = AT{Float64}(undef, n)
        d_out = AT{Float64,2}(undef, (n,n))
        copyto!(d_in, h_in)
        dnrm2_test(device,n)(Val{n}(),d_in,d_out,ndrange=(n,nblk))
        KA.synchronize(device)
        copyto!(h_out, d_out)
        xnorm = norm(h_in, 2)

        @test norm(xnorm .- h_out) <= 1e-10
    end
end

@testset "nrm2" begin
    @kernel function nrm2_test(::Val{n}, d_A, d_out) where {n}
        tx = @index(Local, Linear)
        bx = @index(Group, Linear)

        wa = @localmem Float64 (n,)
        A = @localmem Float64 (n,n)
        for i in 1:n
            A[tx,i] = d_A[tx,i]
        end
        @synchronize

        ExaTron.nrm2!(wa, A, n, tx)
        if bx == 1
            d_out[tx] = wa[tx]
        end
        @synchronize
    end

    @inbounds for i=1:itermax
        L = tril(rand(n,n))
        A = L*transpose(L)
        A .= tril(A) .+ (transpose(tril(A)) .- Diagonal(A))
        wa = zeros(n)
        tron_A = ExaTron.TronDenseMatrix{Array{Float64,2}}(n)
        tron_A.vals .= A
        ExaTron.nrm2!(wa, tron_A, n)

        d_A = AT{Float64,2}(undef, (n,n))
        d_out = AT{Float64}(undef, n)
        h_wa = zeros(n)
        copyto!(d_A, A)
        nrm2_test(device,n)(Val{n}(),d_A,d_out,ndrange=(n,nblk))
        KA.synchronize(device)
        copyto!(h_wa, d_out)

        @test norm(wa .- h_wa) <= 1e-10
    end
end

@testset "dcopy" begin
    @kernel function dcopy_test(::Val{n}, d_in,
                        d_out
                        ) where {n}
        tx = @index(Local, Linear)
        bx = @index(Group, Linear)

        x = @localmem Float64 (n,)
        y = @localmem Float64 (n,)

        x[tx] = d_in[tx]
        @synchronize

        ExaTron.dcopy(n, x, 1, y, 1, tx)

        if bx == 1
            d_out[tx] = y[tx]
        end
        @synchronize

    end

    @inbounds for i=1:itermax
        h_in = rand(n)
        h_out = zeros(n)
        d_in = AT{Float64}(undef, n)
        d_out = AT{Float64}(undef, n)
        copyto!(d_in, h_in)
        dcopy_test(device,n)(Val{n}(),d_in,d_out,ndrange=(n,nblk))
        KA.synchronize(device)
        copyto!(h_out, d_out)

        @test !(false in (h_in .== h_out))
    end
end

@testset "ddot" begin
    @kernel function ddot_test(::Val{n}, d_in,
                        d_out
                        ) where {n}
        tx = @index(Local, Linear)
        bx = @index(Group, Linear)

        x = @localmem Float64 (n,)
        y = @localmem Float64 (n,)
        x[tx] = d_in[tx]
        y[tx] = d_in[tx]
        @synchronize

        v = ExaTron.ddot(n, x, 1, y, 1, tx)

        if bx == 1
            for i in 1:n
                d_out[tx,i] = v
            end
        end
        @synchronize

    end

    @inbounds for i=1:itermax
        h_in = rand(n)
        h_out = zeros((n,n))
        d_in = AT{Float64}(undef, n)
        d_out = AT{Float64,2}(undef, (n,n))
        copyto!(d_in, h_in)
        ddot_test(device, n)(Val{n}(),d_in,d_out,ndrange=(n,nblk))
        KA.synchronize(device)
        copyto!(h_out, d_out)

        @test norm(dot(h_in,h_in) .- h_out, 2) <= 1e-10
    end
end

@testset "dscal" begin
    @kernel function dscal_test(::Val{n}, da::Float64,
                        d_in,
                        d_out
                        ) where {n}
        tx = @index(Local, Linear)
        bx = @index(Group, Linear)

        x = @localmem Float64 (n,)
        if tx <= n
            x[tx] = d_in[tx]
        end
        @synchronize

        ExaTron.dscal(n, da, x, 1, tx)
        if bx == 1
            d_out[tx] = x[tx]
        end
        @synchronize

    end

    for i=1:itermax
        h_in = rand(n)
        h_out = zeros(n)
        da = rand(1)[1]
        d_in = AT{Float64}(undef, n)
        d_out = AT{Float64}(undef, n)
        copyto!(d_in, h_in)
        dscal_test(device,n)(Val{n}(),da,d_in,d_out,ndrange=(n,nblk))
        KA.synchronize(device)
        copyto!(h_out, d_out)

        @test norm(h_out .- (da.*h_in)) <= 1e-12
    end
end

@testset "dtrqsol" begin
    @kernel function dtrqsol_test(::Val{n}, d_x,
                            d_p,
                            d_out,
                            delta::Float64
                            ) where {n}
        tx = @index(Local, Linear)
        bx = @index(Group, Linear)

        x = @localmem Float64 (n,)
        p = @localmem Float64 (n,)

        x[tx] = d_x[tx]
        p[tx] = d_p[tx]
        @synchronize

        sigma = ExaTron.dtrqsol(n, x, p, delta, tx)
        if bx == 1
            for i in 1:n
                d_out[tx,i] = sigma
            end
        end
        @synchronize
    end

    for i=1:itermax
        x = rand(n)
        p = rand(n)
        sigma = abs(rand(1)[1])
        delta = norm(x .+ sigma.*p)

        d_x = AT{Float64}(undef, n)
        d_p = AT{Float64}(undef, n)
        d_out = AT{Float64,2}(undef, (n,n))
        copyto!(d_x, x)
        copyto!(d_p, p)
        dtrqsol_test(device,n)(Val{n}(),d_x,d_p,d_out,delta,ndrange=(n,nblk))
        KA.synchronize(device)

        d_out = d_out |> Array
        @test norm(sigma .- d_out) <= 1e-10
    end
end

@testset "dspcg" begin
    @kernel function dspcg_test(::Val{n}, delta::Float64, rtol::Float64,
                        cg_itermax::Int, dx,
                        dxl,
                        dxu,
                        dA,
                        dg,
                        ds,
                        d_out
                        ) where {n}
        tx = @index(Local, Linear)
        bx = @index(Group, Linear)

        x = @localmem Float64 (n,)
        xl = @localmem Float64 (n,)
        xu = @localmem Float64 (n,)
        g = @localmem Float64 (n,)
        s = @localmem Float64 (n,)
        w = @localmem Float64 (n,)
        wa1 = @localmem Float64 (n,)
        wa2 = @localmem Float64 (n,)
        wa3 = @localmem Float64 (n,)
        wa4 = @localmem Float64 (n,)
        wa5 = @localmem Float64 (n,)
        gfree = @localmem Float64 (n,)
        indfree = @localmem Int (n,)
        iwa = @localmem Int (n,)

        A = @localmem Float64 (n,n)
        B = @localmem Float64 (n,n)
        L = @localmem Float64 (n,n)

        for i in 1:n
            A[i,tx] = dA[i,tx]
        end
        x[tx] = dx[tx]
        xl[tx] = dxl[tx]
        xu[tx] = dxu[tx]
        g[tx] = dg[tx]
        s[tx] = ds[tx]
        @synchronize

        ExaTron.dspcg(n, delta, rtol, cg_itermax, x, xl, xu,
                        A, g, s, B, L, indfree, gfree, w, iwa,
                        wa1, wa2, wa3, wa4, wa5, tx)

        if bx == 1
            d_out[tx] = x[tx]
        end
        @synchronize

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

        dx = AT{Float64}(undef, n)
        dxl = AT{Float64}(undef, n)
        dxu = AT{Float64}(undef, n)
        dA = AT{Float64,2}(undef, (n,n))
        dg = AT{Float64}(undef, n)
        ds = AT{Float64}(undef, n)
        d_out = AT{Float64}(undef, n)

        copyto!(dx, x)
        copyto!(dxl, xl)
        copyto!(dxu, xu)
        copyto!(dA, tron_A.vals)
        copyto!(dg, g)
        copyto!(ds, s)

        dspcg_test(device, n)(Val{n}(),delta,rtol,cg_itermax,dx,dxl,dxu,dA,dg,ds,d_out,ndrange=(n,1))
        KA.synchronize(device)
        h_x = zeros(n)
        copyto!(h_x, d_out)

        ExaTron.dspcg(n, x, xl, xu, tron_A, g, delta, rtol, s, 5, cg_itermax,
                        tron_B, tron_L, indfree, gfree, w, wa, iwa)

        @test norm(x .- h_x) <= 1e-10
    end
end

@testset "dgpnorm" begin
    @kernel function dgpnorm_test(::Val{n}, dx, dxl, dxu, dg, d_out) where {n}
        tx = @index(Local, Linear)
        bx = @index(Group, Linear)

        x = @localmem Float64 (n,)
        xl = @localmem Float64 (n,)
        xu = @localmem Float64 (n,)
        g = @localmem Float64 (n,)

        x[tx] = dx[tx]
        xl[tx] = dxl[tx]
        xu[tx] = dxu[tx]
        g[tx] = dg[tx]
        @synchronize

        v = ExaTron.dgpnorm(n, x, xl, xu, g, tx)
        if bx == 1
            d_out[tx] = v
        end
        @synchronize

    end

    for i=1:itermax
        x = rand(n)
        xl = x .- abs.(rand(n))
        xu = x .+ abs.(rand(n))
        g = 2.0*rand(n) .- 1.0

        dx = AT{Float64}(undef, n)
        dxl = AT{Float64}(undef, n)
        dxu = AT{Float64}(undef, n)
        dg = AT{Float64}(undef, n)
        d_out = AT{Float64}(undef, n)

        copyto!(dx, x)
        copyto!(dxl, xl)
        copyto!(dxu, xu)
        copyto!(dg, g)

        dgpnorm_test(device, n)(Val{n}(), dx, dxl, dxu, dg, d_out, ndrange=(n,n*nblk))
        KA.synchronize(device)
        h_v = zeros(n)
        copyto!(h_v, d_out)

        v = ExaTron.dgpnorm(n, x, xl, xu, g)
        @test norm(h_v .- v) <= 1e-10
    end
end

@testset "dtron" begin
    @kernel function dtron_test(::Val{n}, f::Float64, frtol::Float64, fatol::Float64, fmin::Float64,
                        cgtol::Float64, cg_itermax::Int, delta::Float64, task::Int,
                        disave, ddsave,
                        dx, dxl,
                        dxu, dA,
                        dg, d_out
                        ) where {n}
        tx = @index(Local, Linear)
        bx = @index(Group, Linear)

        x = @localmem Float64 (n,)
        xl = @localmem Float64 (n,)
        xu = @localmem Float64 (n,)
        g = @localmem Float64 (n,)
        xc = @localmem Float64 (n,)
        s = @localmem Float64 (n,)
        wa = @localmem Float64 (n,)
        wa1 = @localmem Float64 (n,)
        wa2 = @localmem Float64 (n,)
        wa3 = @localmem Float64 (n,)
        wa4 = @localmem Float64 (n,)
        wa5 = @localmem Float64 (n,)
        gfree = @localmem Float64 (n,)
        indfree = @localmem Int (n,)
        iwa = @localmem Int (2*n,)

        A = @localmem Float64 (n,n)
        B = @localmem Float64 (n,n)
        L = @localmem Float64 (n,n)

        for i in 1:n
            A[i,tx] = dA[i,tx]
        end
        x[tx] = dx[tx]
        xl[tx] = dxl[tx]
        xu[tx] = dxu[tx]
        g[tx] = dg[tx]
        @synchronize

        ExaTron.dtron(n, x, xl, xu, f, g, A, frtol, fatol, fmin, cgtol,
                        cg_itermax, delta, task, B, L, xc, s, indfree, gfree,
                        disave, ddsave, wa, iwa, wa1, wa2, wa3, wa4, wa5, tx)
        if bx == 1
            d_out[tx] = x[tx]
        end
        @synchronize

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

        dx = AT{Float64}(undef, n)
        dxl = AT{Float64}(undef, n)
        dxu = AT{Float64}(undef, n)
        dA = AT{Float64,2}(undef, (n,n))
        dg = AT{Float64}(undef, n)
        disave = AT{Int}(undef, n)
        ddsave = AT{Float64}(undef, n)
        d_out = AT{Float64}(undef, n)

        copyto!(dx, x)
        copyto!(dxl, xl)
        copyto!(dxu, xu)
        copyto!(dA, tron_A.vals)
        copyto!(dg, g)

        dtron_test(device,n)(Val{n}(),f,frtol,fatol,fmin,cgtol,cg_itermax,delta,task,disave,ddsave,dx,dxl,dxu,dA,dg,d_out,ndrange=(n,n*nblk))
        KA.synchronize(device)
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
    @inline function eval_f(n, x, dA, dc)
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
        @synchronize
        return f
    end

    @inline function eval_g(n, x, g, dA, dc)
        @inbounds for i=1:n
            gval = 0
            @inbounds for j=1:n
                gval += dA[i,j]*x[j]
            end
            g[i] = gval + dc[i]
        end
        @synchronize
        return
    end

    @inline function eval_h(n, scale, x, A, dA, tx)
        for i in 1:n
            A[i,tx] = dA[i,tx]
        end
        @synchronize
        return
    end

    @inline function driver_kernel(n, max_feval::Int, max_minor::Int,
                           x, xl,
                           xu, dA,
                           dc,
                           tx)
        # We start with a shared memory allocation.
        # The first 3*n*sizeof(Float64) bytes are used for x, xl, and xu.
        g = @localmem Float64 (n,)
        xc = @localmem Float64 (n,)
        s = @localmem Float64 (n,)
        wa = @localmem Float64 (n,)
        wa1 = @localmem Float64 (n,)
        wa2 = @localmem Float64 (n,)
        wa3 = @localmem Float64 (n,)
        wa4 = @localmem Float64 (n,)
        wa5 = @localmem Float64 (n,)
        gfree = @localmem Float64 (n,)
        dsave = @localmem Float64 (n,)
        indfree = @localmem Int (n,)
        iwa = @localmem Int (2*n,)
        isave = @localmem Int (n,)

        A = @localmem Float64 (n,n)
        B = @localmem Float64 (n,n)
        L = @localmem Float64 (n,n)

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
                eval_h(n, 1.0, x, A, dA, tx)
                ngev += 1
                nhev += 1
                minor_iter += 1
            end

            # Initialize the trust region bound.

            if task == 0
                gnorm0 = ExaTron.dnrm2(n, g, 1, tx)
                delta = gnorm0
            end

            # Call Tron.

            if search
                delta, task = ExaTron.dtron(n, x, xl, xu, f, g, A, frtol, fatol, fmin, cgtol,
                                            cg_itermax, delta, task, B, L, xc, s, indfree, gfree,
                                            isave, dsave, wa, iwa, wa1, wa2, wa3, wa4, wa5, tx)
            end

            # [3] NEWX: a new point was computed.

            if task == 3
                gnorm_inf = ExaTron.dgpnorm(n, x, xl, xu, g, tx)
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

    @kernel function driver_kernel_test(::Val{n}, max_feval, max_minor,
                                dx, dxl, dxu, dA, dc, d_out) where {n}
        tx = @index(Local, Linear)
        bx = @index(Group, Linear)

        x = @localmem Float64 (n,)
        xl = @localmem Float64 (n,)
        xu = @localmem Float64 (n,)

        x[tx] = dx[tx]
        xl[tx] = dxl[tx]
        xu[tx] = dxu[tx]
        @synchronize

        status, minor_iter = driver_kernel(n, max_feval, max_minor, x, xl, xu, dA, dc,tx)

        if bx == 1
            d_out[tx] = x[tx]
        end
        @synchronize
    end

    max_feval = 500
    max_minor = 100

    tron_A = ExaTron.TronDenseMatrix{Array{Float64,2}}(n)
    tron_B = ExaTron.TronDenseMatrix{Array{Float64,2}}(n)
    tron_L = ExaTron.TronDenseMatrix{Array{Float64,2}}(n)

    dx = AT{Float64}(undef, n)
    dxl = AT{Float64}(undef, n)
    dxu = AT{Float64}(undef, n)
    dA = AT{Float64,2}(undef, (n,n))
    dc = AT{Float64}(undef, n)
    d_out = AT{Float64}(undef, n)

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

        driver_kernel_test(device,n)(Val{n}(),max_feval,max_minor,dx,dxl,dxu,dA,dc,d_out,ndrange=(n,nblk))
        KA.synchronize(device)
        h_x = zeros(n)
        copyto!(h_x, d_out)

        @test norm(h_x .- tron.x) <= 1e-10
    end
end
