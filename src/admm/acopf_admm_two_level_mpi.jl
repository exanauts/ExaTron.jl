function admm_two_level_mpi(case::String;
    outer_iterlim=10, inner_iterlim=800, rho_pq=400.0, rho_va=40000.0, scale=1e-4,
    use_gpu=false, use_linelimit=false, outer_eps=2*1e-4, solve_pf=false, gpu_no=0,
    comm::MPI.Comm=MPI.COMM_WORLD, verbose=1)

    T = Float64; TD = Array{Float64,1}; TI = Array{Int,1}; TM = Array{Float64,2}
    if use_gpu
        if !MPI.has_cuda()
            error("MPI is not compatible with CUDA.")
        end
        CUDA.device!(gpu_no)
        TD = CuArray{Float64,1}; TI = CuArray{Int,1}; TM = CuArray{Float64,2}
    end

    env = AdmmEnv{T,TD,TI,TM}(case, rho_pq, rho_va; use_gpu=use_gpu, use_linelimit=use_linelimit,
            use_twolevel=true, use_mpi=true, solve_pf=solve_pf, gpu_no=gpu_no, verbose=verbose, comm=comm)
    env.params.outer_eps = outer_eps
    mod = Model{T,TD,TI,TM}(env)

    if use_linelimit
        if use_gpu
            # Set rateA in membuf.
            CUDA.@sync @cuda threads=64 blocks=(div(mod.nline-1, 64)+1) set_rateA_kernel(mod.nline, mod.membuf, mod.rateA)
        else
            mod.membuf[29,:] .= mod.rateA
        end
    end

#=
    if use_gpu
        CUDA.device!(gpu_no)

        env = AdmmEnv{Float64, CuArray{Float64,1}, CuArray{Int,1}, CuArray{Float64,2}}(
            case, rho_pq, rho_va; use_gpu=use_gpu, use_linelimit=use_linelimit, use_twolevel=true,
            solve_pf=solve_pf, gpu_no=gpu_no, verbose=verbose, comm=comm,
        )
        env.params.outer_eps = outer_eps
        mod = Model{Float64, CuArray{Float64,1}, CuArray{Int,1}, CuArray{Float64,2}}(env)
        if use_linelimit
            # Set rateA in membuf.
            @cuda threads=64 blocks=(div(mod.nline-1, 64)+1) set_rateA_kernel(mod.nline, mod.membuf, mod.rateA)
        end
    else
        env = AdmmEnv{Float64, Array{Float64,1}, Array{Int,1}, Array{Float64,2}}(
            case, rho_pq, rho_va; use_gpu=use_gpu, use_linelimit=use_linelimit, use_twolevel=true,
            solve_pf=solve_pf, gpu_no=gpu_no, verbose=verbose, comm=comm,
        )
        env.params.outer_eps = outer_eps
        mod = Model{Float64, Array{Float64,1}, Array{Int,1}, Array{Float64,2}}(env)
        if use_linelimit
            mod.membuf[29,:] .= mod.rateA
        end
    end
=#
    admm_restart_two_level_mpi(env, mod; outer_iterlim=outer_iterlim, inner_iterlim=inner_iterlim, scale=scale)
    return env, mod
end