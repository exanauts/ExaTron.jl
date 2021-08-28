function admm_multiperiod_two_level(
    case_prefix::String, load_prefix::String, horizon_length::Int;
    outer_iterlim::Int=20, inner_iterlim::Int=800,
    rho_pq::Float64=400.0, rho_va::Float64=40000.0, scale::Float64=1e-4,
    use_gpu::Bool=false, use_linelimit::Bool=false, gpu_no::Int=0, verbose::Int=1)

    if use_gpu
        CUDA.device!(gpu_no)
        env = AdmmEnv{Float64, CuArray{Float64,1}, CuArray{Int,1}, CuArray{Float64,2}}(
            case_prefix, rho_pq, rho_va;
            use_gpu=true, use_linelimit=use_linelimit, use_twolevel=true, verbose=verbose,
            gpu_no=gpu_no, horizon_length=horizon_length, load_prefix=load_prefix)
    else
        env = AdmmEnv{Float64, Array{Float64,1}, Array{Int,1}, Array{Float64,2}}(
            case_prefix, rho_pq, rho_va;
            use_gpu=false, use_linelimit=use_linelimit, use_twolevel=true, verbose=verbose,
            horizon_length=horizon_length, load_prefix=load_prefix)
    end

    env.params.outer_iterlim = outer_iterlim
    env.params.inner_iterlim = inner_iterlim
    env.params.scale = scale

    T = Float64; TD = Array{Float64,1}; TI = Array{Int,1}; TM = Array{Float64,2}
    if use_gpu
        TD = CuArray{Float64,1}; TI = CuArray{Int,1}; TM = CuArray{Float64,2}
    end

    models = [ ModelWithRamping{T,TD,TI,TM}(env, t) for t=1:horizon_length ]
    admm_multiperiod_restart_two_level(env, models)

    return env, models
end