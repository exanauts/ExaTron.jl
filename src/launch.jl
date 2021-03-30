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
        for k=1:10
            t += ExaTron.launch_kernel(j, nblk; thread_conf=thread_conf)
        end
        t /= 10
        @printf("  average time = %.5f\n", t)
    end
end