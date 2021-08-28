function init_solution!(
    model::ModelWithRamping{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    rho_pq::Float64
)
    inner = model.inner
    inner_sol = inner.solution
    sol = model.ramping_solution

    fill!(sol, 0.0)
    sol.xbar_curr[1:inner.ngen] .= 0.5.*(inner.pgmin[1:inner.ngen] .+ inner.pgmax[1:inner.ngen])
    sol.xbar_tm1_curr[1:inner.ngen] .= 0.5.*(inner.pgmin[1:inner.ngen] .+ inner.pgmax[1:inner.ngen])
    sol.rho .= rho_pq
end

function set_ramping_solution_kernel(
    gen_start::Int, n::Int,
    xbar_ramp::CuDeviceArray{Float64,1}, xbar_inner::CuDeviceArray{Float64,1})
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if tx <= n
        xbar_ramp[tx] = xbar_inner[gen_start + 2*(tx-1)]
    end
    return
end

function set_ramping_solution!(
    curr_model::ModelWithRamping{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}}
)
    inner = curr_model.inner
    inner_sol = inner.solution
    sol = curr_model.ramping_solution

    @cuda threads=64 blocks=(div(inner.ngen-1,64)+1) set_ramping_solution_kernel(
        inner.gen_start, inner.ngen, sol.xbar_curr, inner_sol.xbar_curr
    )
    CUDA.synchronize()
    return
end

function set_ramping_solution!(
    curr_model::ModelWithRamping{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    prev_model::ModelWithRamping{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}}
)
    set_ramping_solution!(curr_model)

    inner = prev_model.inner
    inner_sol = inner.solution
    sol = curr_model.ramping_solution

    @cuda threads=64 blocks=(div(inner.ngen-1,64)+1) set_ramping_solution_kernel(
        inner.gen_start, inner.ngen, sol.xbar_tm1_curr, inner_sol.xbar_curr
    )
    CUDA.synchronize()
    return
end

function update_xbar_ramping_kernel(n::Int, xbar, x_tp1, z_tp1, l_tp1, rho_tp1, x_inner, z, l, rho)
    g = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if g <= n
        @inbounds begin
            xbar[g] = (l_tp1[3*g-2] + rho_tp1[3*g-2]*(x_tp1[2*g-1] + z_tp1[3*g-2])
                     + l[3*g] + rho[3*g]*(x_inner[2*g-1] + z[3*g])) / (rho_tp1[3*g-2] + rho[3*g])
        end
    end

    return
end

function update_z_ramping_kernel(n::Int, z, x, l, rho, xbar_tm1, x_inner, lz, beta)
    g = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if g <= n
        @inbounds begin
            i = 3*g-2
            z[i] = -(lz[i] + l[i] + rho[i]*(x[2*g-1] - xbar_tm1[g])) / (beta + rho[i])
            z[i+1] = -(lz[i+1] + l[i+1] + rho[i+1]*(x_inner[2*g-1] - x[2*g-1] - x[2*g])) / (beta + rho[i+1])
        end
    end

    return
end

function update_z_pgtilde_ramping_kernel(n::Int, z, l, rho, xbar, x_inner, lz, beta)
    g = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if g <= n
        @inbounds begin
            i = 3*g
            z[i] = -(lz[i] + l[i] + rho[i]*(x_inner[2*g-1] - xbar[g])) / (beta + rho[i])
        end
    end

    return
end

function update_l_ramping_kernel(n::Int, l, z, lz, beta)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds begin
            l[i] = -(lz[i] + beta*z[i])
        end
    end

    return
end

function update_rp_ramping_kernel(n::Int, rp, x, z, xbar, xbar_tm1, x_inner, t, T)
    g = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if g <= n
        i = 3*g-2
        if t > 1
            rp[i] = x[2*g-1] - xbar_tm1[g] + z[i]
            rp[i+1] = x_inner[2*g-1] - x[2*g-1] - x[2*g] + z[i+1]
        end

        if t < T
            rp[i+2] = x_inner[2*g-1] - xbar[g] + z[i+2]
        end
    end

    return
end

function admm_multiperiod_solve_single_period(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    ramp_model::ModelWithRamping{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}}
)
    par = env.params
    mod = ramp_model.inner
    sol = mod.solution
    ramp_sol = ramp_model.ramping_solution

    x_curr = sol.x_curr
    xbar_curr = sol.xbar_curr
    z_outer = sol.z_outer
    z_curr = sol.z_curr
    z_prev = sol.z_prev
    l_curr = sol.l_curr
    lz = sol.lz
    rho = sol.rho
    rp = sol.rp
    rd = sol.rd
    rp_old = sol.rp_old
    Ax_plus_By = sol.Ax_plus_By

    u_curr = view(x_curr, 1:mod.nvar_u)
    v_curr = view(x_curr, mod.nvar_u+1:mod.nvar)
    zu_curr = view(z_curr, 1:mod.nvar_u)
    zv_curr = view(z_curr, mod.nvar_u+1:mod.nvar)
    lu_curr = view(l_curr, 1:mod.nvar_u)
    lv_curr = view(l_curr, mod.nvar_u+1:mod.nvar)
    lz_u = view(lz, 1:mod.nvar_u)
    lz_v = view(lz, mod.nvar_u+1:mod.nvar)
    rho_u = view(rho, 1:mod.nvar_u)
    rho_v = view(rho, mod.nvar_u+1:mod.nvar)
    rp_u = view(rp, 1:mod.nvar_u)
    rp_v = view(rp, mod.nvar_u+1:mod.nvar)

    nblk_bus = div(mod.nbus, 32, RoundUp)
    shmem_size = sizeof(Float64)*(14*mod.n+3*mod.n^2) + sizeof(Int)*(4*mod.n)
    shmem_size_gen = sizeof(Float64)*(14*3+3*3^2) + sizeof(Int)*(4*3)

    beta = 1e3
    c = 6.0
    theta = 0.8
    sqrt_d = sqrt(mod.nvar_u + mod.nvar_v)
    OUTER_TOL = sqrt_d*(par.outer_eps)

    shift_lines = 0

    mismatch = Inf
    z_prev_norm = z_curr_norm = Inf

    outer_iter = 0

    while outer_iter < par.outer_iterlim
        outer_iter += 1

        @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) copy_data_kernel(mod.nvar, z_outer, z_curr)
        z_prev_norm = CUDA.norm(z_curr)
        CUDA.synchronize()

        inner_iter = 0
        while inner_iter < par.inner_iterlim
            inner_iter += 1

            @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) copy_data_kernel(mod.nvar, z_prev, z_curr)
            @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) copy_data_kernel(mod.nvar, rp_old, rp)
            CUDA.synchronize()

            if ramp_model.time_index == 1
                @cuda threads=32 blocks=(div(mod.ngen-1,32)+1) generator_kernel_multiperiod_first_gpu(
                        mod.baseMVA, mod.ngen, mod.gen_start,
                        u_curr, xbar_curr, zu_curr, lu_curr, rho_u,
                        ramp_sol.x_curr, ramp_sol.xbar_curr, ramp_sol.z_curr,
                        ramp_sol.l_curr, ramp_sol.rho,
                        mod.pgmin, mod.pgmax, mod.qgmin, mod.qgmax,
                        mod.c2, mod.c1)
            else
                @cuda threads=32 blocks=mod.ngen shmem=shmem_size_gen generator_kernel_multiperiod_rest_gpu(
                        mod.baseMVA, mod.ngen, mod.gen_start,
                        u_curr, xbar_curr, zu_curr, lu_curr, rho_u,
                        ramp_sol.x_curr, ramp_sol.xbar_curr, ramp_sol.xbar_tm1_curr,
                        ramp_sol.z_curr, ramp_sol.l_curr, ramp_sol.rho,
                        ramp_model.gen_membuf,
                        mod.pgmin, mod.pgmax, mod.qgmin, mod.qgmax,
                        mod.c2, mod.c1,
                        ramp_model.ramp_rate)
            end

            @cuda threads=32 blocks=mod.nline shmem=shmem_size polar_kernel_two_level(mod.n, mod.nline, mod.line_start, mod.bus_start, par.scale,
                    u_curr, xbar_curr, zu_curr, lu_curr, rho_u,
                    shift_lines, mod.membuf, mod.YffR, mod.YffI, mod.YftR, mod.YftI,
                    mod.YttR, mod.YttI, mod.YtfR, mod.YtfI, mod.FrBound, mod.ToBound, mod.brBusIdx)

            @cuda threads=32 blocks=nblk_bus bus_kernel_two_level(mod.baseMVA, mod.nbus, mod.gen_start, mod.line_start, mod.bus_start,
                    mod.FrStart, mod.FrIdx, mod.ToStart, mod.ToIdx, mod.GenStart, mod.GenIdx,
                    mod.Pd, mod.Qd, v_curr, xbar_curr, zv_curr, lv_curr, rho_v, mod.YshR, mod.YshI)
            CUDA.synchronize()

            # Update xbar.
            @cuda threads=64 blocks=(div(mod.ngen-1, 64)+1) update_xbar_generator_kernel(mod.ngen, mod.gen_start, u_curr, v_curr,
                                            xbar_curr, zu_curr, zv_curr, lu_curr, lv_curr, rho_u, rho_v)
            @cuda threads=64 blocks=(div(mod.nline-1, 64)+1) update_xbar_branch_kernel(mod.nline, mod.line_start, u_curr, v_curr,
                                            xbar_curr, zu_curr, zv_curr, lu_curr, lv_curr, rho_u, rho_v)
            @cuda threads=64 blocks=(div(mod.nbus-1, 64)+1) update_xbar_bus_kernel(mod.nbus, mod.line_start, mod.bus_start, mod.FrStart, mod.FrIdx, mod.ToStart, mod.ToIdx, u_curr, v_curr, xbar_curr, zu_curr, zv_curr, lu_curr, lv_curr, rho_u, rho_v)
            CUDA.synchronize()

            # Update z.
            @cuda threads=64 blocks=(div(mod.ngen-1, 64)+1) update_zu_generator_kernel(mod.ngen, mod.gen_start, u_curr,
                                            xbar_curr, zu_curr, lu_curr, rho_u, lz_u, beta)
            @cuda threads=64 blocks=(div(mod.nline-1, 64)+1) update_zu_branch_kernel(mod.nline, mod.line_start, mod.bus_start, mod.brBusIdx, u_curr, xbar_curr, zu_curr, lu_curr, rho_u, lz_u, beta)
            @cuda threads=64 blocks=(div(mod.nvar_v-1, 64)+1) update_zv_kernel(mod.nvar_v, v_curr, xbar_curr, zv_curr,
                                            lv_curr, rho_v, lz_v, beta)
            CUDA.synchronize()

            # Update multiiplier and residuals.
            @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) update_l_kernel(mod.nvar, l_curr, z_curr, lz, beta)
            @cuda threads=64 blocks=(div(mod.ngen-1, 64)+1) compute_primal_residual_u_generator_kernel(mod.ngen, mod.gen_start, rp_u, u_curr, xbar_curr, zu_curr)
            @cuda threads=64 blocks=(div(mod.nline-1, 64)+1) compute_primal_residual_u_branch_kernel(mod.nline, mod.line_start, mod.bus_start, mod.brBusIdx, rp_u, u_curr, xbar_curr, zu_curr)
            @cuda threads=64 blocks=(div(mod.nvar_v-1, 64)+1) compute_primal_residual_v_kernel(mod.nvar_v, rp_v, v_curr, xbar_curr, zv_curr)
            @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) vector_difference(mod.nvar, rd, z_curr, z_prev)
            CUDA.synchronize()

            CUDA.@sync @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) vector_difference(mod.nvar, Ax_plus_By, rp, z_curr)
            mismatch = CUDA.norm(Ax_plus_By)

            primres = CUDA.norm(rp)
            dualres = CUDA.norm(rd)
            z_curr_norm = CUDA.norm(z_curr)
            eps_pri = sqrt_d / (2500*outer_iter)

            if par.verbose > 0
                if inner_iter == 1 || (inner_iter % 50) == 0
                    @printf("%8s  %8s  %10s  %10s  %10s  %10s  %10s  %10s\n",
                            "Outer", "Inner", "PrimRes", "EpsPrimRes", "DualRes", "||z||", "||Ax+By||", "Beta")
                end
                @printf("%8d  %8d  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e\n",
                        outer_iter, inner_iter, primres, eps_pri, dualres, z_curr_norm, mismatch, beta)
            end

            if primres <= eps_pri || dualres <= par.DUAL_TOL
                break
            end
        end # while inner loop

        if mismatch <= OUTER_TOL
            break
        end

        CUDA.@sync @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) update_lz_kernel(mod.nvar, par.MAX_MULTIPLIER, z_curr, lz, beta)

        if z_curr_norm > theta*z_prev_norm
            beta = min(c*beta, 1e24)
        end
    end # while outer loop

    return
end

function admm_multiperiod_restart_two_level(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    models::Array{ModelWithRamping{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},1}
)
    par = env.params

    # Solve each single period problem for warm-start.
    for t=1:env.horizon_length
        admm_restart_two_level(env, models[t].inner)
    end

    # Set ramping solution for warm-start.
    set_ramping_solution!(models[1])
    for t=2:env.horizon_length
        set_ramping_solution!(models[t], models[t-1])
    end

    beta = 1e3
    c = 6.0
    theta = 0.8

    nvar_xbar_curr = length(models[1].ramping_solution.xbar_curr)
    nvar_z_curr = length(models[1].ramping_solution.z_curr)

    nvar = models[1].inner.ngen
    for t=2:env.horizon_length
        nvar += 3*models[t].inner.ngen
    end
    sqrt_d = sqrt(nvar)
    OUTER_TOL = sqrt_d*par.outer_eps

    @printf("Starting two-level ADMM for the temporal decomposition . . .\n")

    mismatch = Inf
    z_curr_norm = z_prev_norm = Inf

    outer_iter = 0
    while outer_iter < par.outer_iterlim
        outer_iter += 1

        inner_iter = 0
        while inner_iter < par.inner_iterlim
            inner_iter += 1

            z_prev_norm = 0.0
            for t=1:env.horizon_length
                sol = models[t].ramping_solution
                CUDA.@sync @cuda threads=64 blocks=(div(nvar_z_curr-1, 64)+1) copy_data_kernel(nvar_z_curr, sol.z_prev, sol.z_curr)
                z_prev_norm += CUDA.norm(sol.z_prev)^2
            end
            z_prev_norm = sqrt(z_prev_norm)

            for t=1:env.horizon_length
                # Solve each single period problem.
                admm_multiperiod_solve_single_period(env, models[t])
            end

            # Update the consensus variable xbar for ramping.
            for t=1:env.horizon_length-1
                inner = models[t].inner
                sol = models[t].ramping_solution
                sol_tp1 = models[t+1].ramping_solution
                @cuda threads=64 blocks=(div(inner.ngen-1,64)+1) update_xbar_ramping_kernel(
                    inner.ngen, sol.xbar_curr, sol_tp1.x_curr, sol_tp1.z_curr, sol_tp1.l_curr, sol_tp1.rho,
                    inner.solution.x_curr, sol.z_curr, sol.l_curr, sol.rho)
                #=
                for g=1:inner.ngen
                    sol.xbar_curr[g] =
                       (sol_tp1.l_curr[3*g-2] + sol_tp1.rho[3*g-2]*(sol_tp1.x_curr[2*g-1] + sol_tp1.z_curr[3*g-2])
                        + sol.l_curr[3*g] + sol.rho[3*g]*(inner.solution.x_curr[2*g-1] + sol.z_curr[3*g])
                       ) / (sol_tp1.rho[3*g-2] + sol.rho[3*g])
                end
                =#
            end
            CUDA.synchronize()

            for t=2:env.horizon_length
                @cuda threads=64 blocks=(div(nvar_xbar_curr-1, 64)+1) copy_data_kernel(
                    nvar_xbar_curr, models[t].ramping_solution.xbar_tm1_curr, models[t-1].ramping_solution.xbar_curr)
            end
            CUDA.synchronize()

            # Update z for ramping.
            for t=2:env.horizon_length
                inner = models[t].inner
                sol = models[t].ramping_solution
                @cuda threads=64 blocks=(div(inner.ngen-1,64)+1) update_z_ramping_kernel(
                    inner.ngen, sol.z_curr, sol.x_curr, sol.l_curr, sol.rho,
                    sol.xbar_tm1_curr, inner.solution.x_curr, sol.lz, beta)
                #=
                for g=1:inner.ngen
                    # PG_HAT
                    sol.z_curr[3*g-2] =
                        -(sol.lz[3*g-2] + sol.l_curr[3*g-2] +
                             sol.rho[3*g-2]*(sol.x_curr[2*g-1] - sol.xbar_tm1_curr[g])
                         ) / (beta + sol.rho[3*g-2])
                    # Slack for ramping
                    sol.z_curr[3*g-1] =
                        -(sol.lz[3*g-1] + sol.l_curr[3*g-1] +
                             sol.rho[3*g-1]*(inner.solution.x_curr[2*g-1] - sol.x_curr[2*g-1] - sol.x_curr[2*g])
                         ) / (beta + sol.rho[3*g-1])
                end
                =#
            end

            for t=1:env.horizon_length-1
                inner = models[t].inner
                sol = models[t].ramping_solution
                @cuda threads=64 blocks=(div(inner.ngen-1,64)+1) update_z_pgtilde_ramping_kernel(
                    inner.ngen, sol.z_curr, sol.l_curr, sol.rho, sol.xbar_curr,
                    inner.solution.x_curr, sol.lz, beta)
                #=
                for g=1:inner.ngen
                    # PG_TILDE
                    sol.z_curr[3*g] =
                        -(sol.lz[3*g] + sol.l_curr[3*g] + sol.rho[3*g]*(inner.solution.x_curr[2*g-1] - sol.xbar_curr[g])
                         ) / (beta + sol.rho[3*g])
                end
                =#
            end
            CUDA.synchronize()

            for t=1:env.horizon_length
                sol = models[t].ramping_solution
                @cuda threads=64 blocks=(div(nvar_z_curr-1,64)+1) update_l_ramping_kernel(
                    nvar_z_curr, sol.l_curr, sol.z_curr, sol.lz, beta)
                #sol.l_curr .= -(sol.lz .+ beta.*sol.z_curr)
            end

            for t=1:env.horizon_length
                inner = models[t].inner
                sol = models[t].ramping_solution
                @cuda threads=64 blocks=(div(inner.ngen-1,64)+1) update_rp_ramping_kernel(
                    inner.ngen, sol.rp, sol.x_curr, sol.z_curr, sol.xbar_curr,
                    sol.xbar_tm1_curr, inner.solution.x_curr, t, env.horizon_length)
                #=
                for g=1:inner.ngen
                    if t > 1
                        sol.rp[3*g-2] = sol.x_curr[2*g-1] - sol.xbar_tm1_curr[g] + sol.z_curr[3*g-2]
                        sol.rp[3*g-1] = inner.solution.x_curr[2*g-1] - sol.x_curr[2*g-1] - sol.x_curr[2*g] + sol.z_curr[3*g-1]
                    end

                    if t < env.horizon_length
                        sol.rp[3*g] = inner.solution.x_curr[2*g-1] - sol.xbar_curr[g] + sol.z_curr[3*g]
                    end
                end
                =#
            end
            CUDA.synchronize()

            primres = 0.0
            dualres = 0.0
            mismatch = 0.0
            z_curr_norm = 0.0
            for t=1:env.horizon_length
                sol = models[t].ramping_solution
                primres += CUDA.norm(sol.rp)^2
                CUDA.@sync @cuda threads=64 blocks=(div(nvar_z_curr-1,64)+1) vector_difference(nvar_z_curr, sol.Ax_plus_By, sol.rp, sol.z_curr)
                CUDA.@sync @cuda threads=64 blocks=(div(nvar_z_curr-1,64)+1) vector_difference(nvar_z_curr, sol.rd, sol.z_curr, sol.z_prev)
                #sol.Ax_plus_By .= sol.rp .- sol.z_curr
                mismatch += CUDA.norm(sol.Ax_plus_By)^2
                z_curr_norm += CUDA.norm(sol.z_curr)^2
                #sol.rd .= sol.z_curr .- sol.z_prev
                dualres += CUDA.norm(sol.rd)^2
            end
            primres = sqrt(primres)
            dualres = sqrt(dualres)
            mismatch = sqrt(mismatch)
            z_curr_norm = sqrt(z_curr_norm)
            eps_pri = sqrt_d / (2500*outer_iter)

            if par.verbose > 0
                if inner_iter == 1 || (inner_iter % 50) == 0
                    @printf("[T] %8s  %8s  %10s  %10s  %10s  %10s  %10s  %10s\n",
                            "Outer", "Inner", "PrimRes", "EpsPrimRes", "DualRes", "||z||", "||Ax+By||", "Beta")
                end

                @printf("[T] %8d  %8d  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e\n",
                        outer_iter, inner_iter, primres, eps_pri, dualres, z_curr_norm, mismatch, beta)
            end

            if primres <= eps_pri || dualres <= par.DUAL_TOL
                break
            end
        end # while inner loop

        if mismatch <= OUTER_TOL
            break
        end

        for t=1:env.horizon_length
            sol = models[t].ramping_solution
            @cuda threads=64 blocks=(div(nvar_z_curr-1,64)+1) update_lz_kernel(nvar_z_curr, par.MAX_MULTIPLIER, sol.z_curr, sol.lz, beta)
            #=
            sol.lz .= max.(-par.MAX_MULTIPLIER,
                           min.(par.MAX_MULTIPLIER, sol.lz .+ (beta .* sol.z_curr)))
            =#
        end
        CUDA.synchronize()

        if z_curr_norm > theta*z_prev_norm
            beta = min(c*beta, 1e24)
        end
    end # while outer loop

    objval = 0.0
    for t=1:env.horizon_length-1
        inner = models[t].inner
        sol = models[t].ramping_solution
        objval += sum(inner.c2[g]*(inner.baseMVA*sol.xbar_curr[g])^2 +
                      inner.c1[g]*(inner.baseMVA*sol.xbar_curr[g]) +
                      inner.c0[g] for g=1:inner.ngen)
    end
    inner = models[env.horizon_length].inner
    objval += sum(inner.c2[g]*(inner.baseMVA*inner.solution.xbar_curr[2*g-1])^2 +
                  inner.c1[g]*(inner.baseMVA*inner.solution.xbar_curr[2*g-1]) +
                  inner.c0[g] for g=1:inner.ngen)
    @printf("Objective value = %.6e\n", objval)

    return
end

#=
function admm_multiperiod_two_level_gpu(
    case_prefix::String, load_prefix::String, horizon_length::Int;
    outer_iterlim::Int=20, inner_iterlim::Int=800,
    rho_pq::Float64=400.0, rho_va::Float64=40000.0, scale::Float64=1e-4,
    use_linelimit::Bool=false, verbose::Int=1
)
    env = AdmmEnv{Float64, CuArray{Float64,1}, CuArray{Int,1}, CuArray{Float64,2}}(
        case_prefix, rho_pq, rho_va;
        use_gpu=false, use_linelimit=use_linelimit, use_twolevel=true, verbose=verbose,
        horizon_length=horizon_length, load_prefix=load_prefix
    )

    env.params.outer_iterlim = outer_iterlim
    env.params.inner_iterlim = inner_iterlim
    env.params.scale = scale

    T = Float64; TD = Array{Float64,1}; TI = Array{Int,1}; TM = Array{Float64,2}
    models = [ ModelWithRamping{T,TD,TI,TM}(env, t) for t=1:horizon_length ]

    admm_multiperiod_restart_two_level(env, models)

    return env, models
end
=#