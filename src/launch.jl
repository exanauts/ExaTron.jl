function launch_cpu(n::Int, nblk::Int)
    x = Array{Float64}(undef, (n, nblk))
    xl = Array{Float64}(undef, (n, nblk))
    xu = Array{Float64}(undef, (n, nblk))

    @inbounds for j=1:nblk
        for i=1:n
            x[i,j] = 1.0
            xl[i,j] = -Inf
            xu[i,j] = i
        end
    end

    @inbounds for j=1:nblk
        nele_hess = div(n*(n+1), 2)
        tron = ExaTron.createProblem(n, xl[:,j], xu[:,j], nele_hess, eval_f_cpu, eval_grad_f_cpu, eval_h_cpu;
                                     :matrix_type => :Dense)
        tron.x .= x[:,j]
        status = ExaTron.solveProblem(tron)
        x[:,j] .= tron.x
        for i=1:n
            @assert abs(x[i,j] - i) <= 1e-10
        end
    end
end

function launch_kernel(n::Int, nblk::Int; thread_conf = :one)
    shmem_size = sizeof(Float64)*(14*n+3*n^2) + sizeof(Int)*(4*n)

    x = Array{Float64}(undef, (n, nblk))
    xl = Array{Float64}(undef, (n, nblk))
    xu = Array{Float64}(undef, (n, nblk))

    for j=1:nblk
        for i=1:n
            x[i,j] = 1.0
            xl[i,j] = -Inf
            xu[i,j] = i
        end
    end

    cu_x = CuArray{Float64}(undef, (n, nblk))
    cu_xl = CuArray{Float64}(undef, (n, nblk))
    cu_xu = CuArray{Float64}(undef, (n, nblk))

    copyto!(cu_x, x)
    copyto!(cu_xl, xl)
    copyto!(cu_xu, xu)

    if thread_conf == :one
        tgpu = CUDA.@timed @cuda threads=32 blocks=nblk shmem=shmem_size tron2_kernel(n, 200, 500, 1e-6, cu_x, cu_xl, cu_xu)
    else
        tgpu = CUDA.@timed @cuda threads=(n,n) blocks=nblk shmem=shmem_size tron2_kernel(n, 200, 500, 1e-6, cu_x, cu_xl, cu_xu)
    end

    copyto!(x, cu_x)
    @inbounds for j=1:nblk
        for i=1:n
            @assert abs(x[i,j] - i) <= 1e-10
        end
    end

    return tgpu.time
end

function batch_run_kernel(n::Int, nblk::Int; thread_conf = :one)
    # Warm-up
    launch_kernel(1, 1)

    for j=1:n
        println("n = ", j, " nblk = ", nblk)
        t = 0
        for k=1:50
            t += ExaTron.launch_kernel(j, nblk; thread_conf=thread_conf)
        end
        t /= 50
        @printf("  average time = %.5f\n", t)
    end
end

@inline function cholesky_right(n::Int, L::CuDeviceArray{Float64,2})
    tx = threadIdx().x

    @inbounds for j=1:n
        if L[j,j] <= 0.0
            CUDA.sync_threads()
            return -1
        end

        # Update the jth column.
        Ljj = sqrt(L[j,j])
        if tx >= j && tx <= n
            L[tx,j] /= Ljj
        end
        CUDA.sync_threads()

        # Update the trailing matrix.
        for k=j+1:n
            if tx >= k && tx <= n
                L[tx,k] -= L[tx,j] * L[k,j]
            end
        end
        CUDA.sync_threads()
    end

    #=
    if tx <= n
        @inbounds for j=1:n
            if tx > j
                L[j,tx] = L[tx,j]
            end
        end
    end
    CUDA.sync_threads()
    =#

    return 0
end

@inline function cholesky_left(n::Int, L::CuDeviceArray{Float64,2})
    tx = threadIdx().x

    @inbounds for j=1:n
        # Apply the pending updates.
        if j > 1
            if tx >= j && tx <= n
                for k=1:j-1
                    L[tx,j] -= L[tx,k] * L[j,k]
                end
            end
        end
        CUDA.sync_threads()

        if (L[j,j] <= 0)
            CUDA.sync_threads()
            return -1
        end

        Ljj = sqrt(L[j,j])
        if tx >= j && tx <= n
            L[tx,j] /= Ljj
        end
        CUDA.sync_threads()
    end

    #=
    if tx <= n
        @inbounds for j=1:n
            if tx > j
                L[j,tx] = L[tx,j]
            end
        end
    end
    CUDA.sync_threads()
    =#

    return 0
end

function cholesky_kernel(n::Int, use_left::Bool, _L::CuDeviceArray{Float64,3})
    tx = threadIdx().x
    bx = blockIdx().x

    L = @cuDynamicSharedMem(Float64, (n,n))
    for j=1:n
        if tx <= n
            L[tx,j] = _L[tx,j,bx]
        end
    end
    CUDA.sync_threads()

    if use_left
        cholesky_left(n, L)
    else
        cholesky_right(n, L)
    end

    for j=1:n
        if tx <= n
            _L[tx,j,bx] = L[tx,j]
        end
    end

    CUDA.sync_threads()
end

function batch_cholesky(n::Int, nblk::Int)
    Random.seed!(0)

    for j=1:n
        println("n = ", j, " nblk = ", nblk)

        left = 0.0
        right = 0.0
        for k=1:50
            L = rand(j,j,nblk)
            A = zeros(j,j,nblk)
            for i=1:nblk
                A[:,:,i] .= tril(L[:,:,i])*transpose(L[:,:,i])
                A[:,:,i] .= tril(A[:,:,i]) .+ (transpose(tril(A[:,:,i])) .- Diagonal(A[:,:,i]))
                for ii=1:j
                    A[ii,ii,i] += 50.0
                end
            end

            cu_A = CuArray{Float64}(undef, (j,j,nblk))
            h_L = zeros(j,j,nblk)

            # Warm-up
            copyto!(cu_A, A)
            tgpu = CUDA.@timed @cuda threads=32 blocks=nblk shmem=(j^2*sizeof(Float64)) cholesky_kernel(j, true, cu_A)
            copyto!(cu_A, A)
            tgpu = CUDA.@timed @cuda threads=32 blocks=nblk shmem=(j^2*sizeof(Float64)) cholesky_kernel(j, false, cu_A)

            # Left-looking
            copyto!(cu_A, A)
            tgpu = CUDA.@timed @cuda threads=32 blocks=nblk shmem=(j^2*sizeof(Float64)) cholesky_kernel(j, true, cu_A)
            left += tgpu.time
            copyto!(h_L, cu_A)
            for i=1:nblk
                @test norm(A[:,:,i] .- tril(h_L[:,:,i])*transpose(tril(h_L[:,:,i]))) <= 1e-10
            end

            # Right-looking
            copyto!(cu_A, A)
            tgpu = CUDA.@timed @cuda threads=32 blocks=nblk shmem=(j^2*sizeof(Float64)) cholesky_kernel(j, false, cu_A)
            right += tgpu.time
            copyto!(h_L, cu_A)
            for i=1:nblk
                @test norm(A[:,:,i] .- tril(h_L[:,:,i])*transpose(tril(h_L[:,:,i]))) <= 1e-10
            end
        end

        @printf(" average  left-looking = %.6f\n", left/50)
        @printf(" average right-looking = %.6f\n", right/50)
    end
end


@inline function dtsol_no_upper(n::Int, L::CuDeviceArray{Float64,2},
    r::CuDeviceArray{Float64,1})
    # Solve L'*x = r and store the result in r.
    tx = threadIdx().x

    if tx == 1
        r[n] = r[n]/L[n,n]

        @inbounds for j=n-1:-1:1
            temp = 0.0
            for k=j+1:n
                temp = temp + L[k,j]*r[k]
            end
            r[j] = (r[j] - temp)/L[j,j]
        end
    end
    CUDA.sync_threads()

    return
end

@inline function dtsol_upper(n::Int, L::CuDeviceArray{Float64,2},
    r::CuDeviceArray{Float64,1})
    # Solve L'*x = r and store the result in r.
    tx = threadIdx().x

    @inbounds for j=n:-1:1
        if tx == 1
            r[j] = r[j] / L[j,j]
        end
        CUDA.sync_threads()

        if tx < j
            r[tx] = r[tx] - L[tx,j]*r[j]
        end
        CUDA.sync_threads()
    end

    return
end

function dtsol_kernel(n::Int, use_upper::Bool, howMany::Int,
    _L::CuDeviceArray{Float64,3}, _r::CuDeviceArray{Float64,2})
    tx = threadIdx().x
    bx = blockIdx().x

    L = @cuDynamicSharedMem(Float64, (n,n))
    r = @cuDynamicSharedMem(Float64, n, n^2*sizeof(Float64))

    if tx <= n
        for j=1:n
            L[tx,j] = _L[tx,j,bx]
        end
        r[tx] = _r[tx,bx]
    end
    CUDA.sync_threads()

    if use_upper
        if tx <= n
            @inbounds for j=1:n
                if tx > j
                    L[j,tx] = L[tx,j]
                end
            end
        end
        CUDA.sync_threads()
    end

    for k=1:howMany
        if use_upper
            dtsol_upper(n, L, r)
        else
            dtsol_no_upper(n, L, r)
        end
    end

    if tx <= n
        _r[tx,bx] = r[tx]
    end
    CUDA.sync_threads()
end

function batch_dtsol(n::Int, nblk::Int)
    Random.seed!(0)

    for j=1:n
        println("n = ", j, " nblk = ", nblk)

        r = zeros(j,nblk)
        for i=1:nblk
            for k=1:j
                r[k,i] = k
            end
        end

        howMany = max(1, div(j,2))

        upper = 0.0
        no_upper = 0.0
        for k=1:50
            L = rand(j,j,nblk)
            for i=1:nblk
                tril!(L[:,:,i])
                for ii=1:j
                    L[ii,ii,i] += 10.0
                end
            end
            host_r1 = zeros(j,nblk)
            host_r2 = zeros(j,nblk)
            cu_L = CuArray{Float64}(undef, (j,j,nblk))
            cu_r = CuArray{Float64}(undef, (j,nblk))
            copyto!(cu_L, L)

            # Warm-up
            copyto!(cu_r, r)
            tgpu = CUDA.@timed @cuda threads=32 blocks=nblk shmem=((n^2+n)*sizeof(Float64)) dtsol_kernel(j, true, howMany, cu_L, cu_r)
            copyto!(cu_r, r)
            tgpu = CUDA.@timed @cuda threads=32 blocks=nblk shmem=((n^2+n)*sizeof(Float64)) dtsol_kernel(j, false, howMany, cu_L, cu_r)

            # Use upper
            copyto!(cu_r, r)
            tgpu = CUDA.@timed @cuda threads=32 blocks=nblk shmem=((n^2+n)*sizeof(Float64)) dtsol_kernel(j, true, howMany, cu_L, cu_r)
            upper += tgpu.time
            copyto!(host_r1, cu_r)

            # Do use upper
            copyto!(cu_r, r)
            tgpu = CUDA.@timed @cuda threads=32 blocks=nblk shmem=((n^2+n)*sizeof(Float64)) dtsol_kernel(j, false, howMany, cu_L, cu_r)
            no_upper += tgpu.time
            copyto!(host_r2, cu_r)

            @test norm(host_r1 .- host_r2) <= 1e-10
        end

        @printf("  average upper    = %.6f\n", upper / 50)
        @printf("  average no-upper = %.6f\n", no_upper / 50)
    end
end