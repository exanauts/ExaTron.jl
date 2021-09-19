function acopf_admm_rolling_horizon_two_level(case::String, load_prefix::String;
    outer_iterlim=10, inner_iterlim=800, rho_pq=400.0, rho_va=40000.0, scale=1e-4,
    use_gpu=false, use_linelimit=false, outer_eps=2*1e-4, solve_pf=false, gpu_no=0,
    ramp_ratio=0.2, start_period=1, end_period=6, verbose=1)

    T = Float64; TD = Array{Float64,1}; TI = Array{Int,1}; TM = Array{Float64,2}
    if use_gpu
        CUDA.device!(gpu_no)
        TD = CuArray{Float64,1}; TI = CuArray{Int,1}; TM = CuArray{Float64,2}
    end

    env = AdmmEnv{T,TD,TI,TM}(case, rho_pq, rho_va; use_gpu=use_gpu, use_linelimit=use_linelimit,
            use_twolevel=true, load_prefix=load_prefix, solve_pf=solve_pf, gpu_no=gpu_no, verbose=verbose)
    env.params.outer_eps = outer_eps
    env.params.outer_iterlim = outer_iterlim
    env.params.inner_iterlim = inner_iterlim
    mod = Model{T,TD,TI,TM}(env; ramp_ratio=ramp_ratio)

    if use_linelimit
        if use_gpu
            # Set rateA in membuf.
            @cuda threads=64 blocks=(div(mod.nline-1, 64)+1) set_rateA_kernel(mod.nline, mod.membuf, mod.rateA)
        else
            mod.membuf[29,:] .= (mod.rateA ./ mod.baseMVA).^2
        end
    end

    admm_restart_rolling_horizon_two_level(env, mod; start_period=start_period, end_period=end_period)
    return env, mod
end