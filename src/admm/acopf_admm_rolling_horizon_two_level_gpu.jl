function update_real_power_current_bounds(
    ngen::Int, gen_start::Int,
    pgmin_curr::CuDeviceArray{Float64,1}, pgmax_curr::CuDeviceArray{Float64,1},
    pgmin_orig::CuDeviceArray{Float64,1}, pgmax_orig::CuDeviceArray{Float64,1},
    ramp_rate::CuDeviceArray{Float64,1}, x_curr::CuDeviceArray{Float64,1})

    g = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if g <= ngen
        pg_idx = gen_start + 2*(g-1)
        pgmin_curr[g] = max(pgmin_orig[g], x_curr[pg_idx] - ramp_rate[g])
        pgmax_curr[g] = min(pgmax_orig[g], x_curr[pg_idx] + ramp_rate[g])
    end
    return
end

function admm_restart_rolling_horizon_two_level(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::Model{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}};
    start_period=1, end_period=6)

    @assert env.load_specified == true
    @assert start_period >= 1 && end_period <= size(env.load.pd,2)

    par = env.params
    sol = mod.solution
    nblk_gen = div(mod.ngen-1, 64) + 1

    for t=start_period:end_period
        mod.Pd = env.load.pd[:,t]
        mod.Qd = env.load.qd[:,t]
        is_solved = admm_restart_two_level(env, mod; outer_iterlim=par.outer_iterlim, inner_iterlim=par.inner_iterlim)
        if !is_solved
            @printf("Solve failed for time period %d. Resolving it from a different starting point . . .\n", t)
            is_solved = admm_restart_two_level(env, mod; outer_iterlim=par.outer_iterlim, inner_iterlim=par.inner_iterlim)
            if !is_solved
                @printf("Failed again time period %d . . .\n", t)
                break
            end
        end
        CUDA.@sync @cuda threads=64 blocks=nblk_gen update_real_power_current_bounds(mod.ngen, mod.gen_start,
            mod.pgmin_curr, mod.pgmax_curr, mod.pgmin, mod.pgmax,
            mod.ramp_rate, mod.solution.x_curr)
    end
end