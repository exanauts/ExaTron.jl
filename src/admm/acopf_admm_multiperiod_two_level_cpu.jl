function init_solution!(
    model::ModelWithRamping{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    rho_pq::Float64
)
    inner = model.inner
    inner_sol = inner.solution
    sol = model.ramping_solution

    fill!(sol, 0.0)
    for g=1:inner.ngen
        sol.xbar_curr[g] = 0.5*(inner.pgmin[g] + inner.pgmax[g])
        sol.xbar_tm1_curr[g] = 0.5*(inner.pgmin[g] + inner.pgmax[g])
    end
    sol.rho .= rho_pq
end

function set_ramping_solution!(
    curr_model::ModelWithRamping{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    inner = curr_model.inner
    inner_sol = inner.solution
    sol = curr_model.ramping_solution

    for g=1:inner.ngen
        pg_idx = inner.gen_start + 2*(g-1)
        sol.xbar_curr[g] = inner_sol.xbar_curr[pg_idx]
    end

    return
end

function set_ramping_solution!(
    curr_model::ModelWithRamping{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    prev_model::ModelWithRamping{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    set_ramping_solution!(curr_model)

    inner = prev_model.inner
    inner_sol = inner.solution
    sol = curr_model.ramping_solution

    for g=1:inner.ngen
        pg_idx = inner.gen_start + 2*(g-1)
        sol.xbar_tm1_curr[g] = inner_sol.xbar_curr[pg_idx]
    end
end

function admm_multiperiod_solve_single_period(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    ramp_model::ModelWithRamping{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
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

        z_outer .= z_curr
        z_prev_norm = norm(z_outer)

        inner_iter = 0
        while inner_iter < par.inner_iterlim
            inner_iter += 1

            z_prev .= z_curr
            rp_old .= rp

            if ramp_model.time_index == 1
                generator_kernel_multiperiod_first_cpu(
                    mod.baseMVA, mod.ngen, mod.gen_start,
                    u_curr, xbar_curr, zu_curr, lu_curr, rho_u,
                    ramp_sol.x_curr, ramp_sol.xbar_curr, ramp_sol.z_curr,
                    ramp_sol.l_curr, ramp_sol.rho,
                    mod.pgmin, mod.pgmax, mod.qgmin, mod.qgmax,
                    mod.c2, mod.c1)
            else
                generator_kernel_multiperiod_rest_cpu(
                    mod.baseMVA, mod.ngen, mod.gen_start,
                    u_curr, xbar_curr, zu_curr, lu_curr, rho_u,
                    ramp_sol.x_curr, ramp_sol.xbar_curr, ramp_sol.xbar_tm1_curr,
                    ramp_sol.z_curr, ramp_sol.l_curr, ramp_sol.rho,
                    ramp_model.gen_membuf,
                    mod.pgmin, mod.pgmax, mod.qgmin, mod.qgmax,
                    mod.c2, mod.c1,
                    ramp_model.ramp_rate)
            end

            polar_kernel_two_level_cpu(mod.n, mod.nline, mod.line_start, mod.bus_start, par.scale,
                u_curr, xbar_curr, zu_curr, lu_curr, rho_u,
                shift_lines, mod.membuf, mod.YffR, mod.YffI, mod.YftR, mod.YftI,
                mod.YttR, mod.YttI, mod.YtfR, mod.YtfI, mod.FrBound, mod.ToBound, mod.brBusIdx)

            bus_kernel_two_level_cpu(mod.baseMVA, mod.nbus, mod.gen_start, mod.line_start, mod.bus_start,
                mod.FrStart, mod.FrIdx, mod.ToStart, mod.ToIdx, mod.GenStart, mod.GenIdx,
                mod.Pd, mod.Qd, v_curr, xbar_curr, zv_curr, lv_curr, rho_v, mod.YshR, mod.YshI)

            update_xbar(mod, u_curr, v_curr, xbar_curr, zu_curr, zv_curr, lu_curr, lv_curr, rho_u, rho_v)
            update_zu(mod, u_curr, xbar_curr, zu_curr, lu_curr, rho_u, lz_u, beta)
            zv_curr .= (-(lz_v .+ lv_curr .+ rho_v.*(v_curr .- xbar_curr))) ./ (beta .+ rho_v)

            l_curr .= -(lz .+ beta.*z_curr)

            compute_primal_residual_u(mod, rp_u, u_curr, xbar_curr, zu_curr)
            rp_v .= v_curr .- xbar_curr .+ zv_curr

            rd .= z_curr .- z_prev
            Ax_plus_By .= rp .- z_curr

            primres = norm(rp)
            dualres = norm(rd)
            z_curr_norm = norm(z_curr)
            mismatch = norm(Ax_plus_By)
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
        end # inner while loop

        if mismatch <= OUTER_TOL
            break
        end

        lz .= max.(-par.MAX_MULTIPLIER, min.(par.MAX_MULTIPLIER, lz .+ (beta .* z_curr)))

        if z_curr_norm > theta*z_prev_norm
            beta = min(c*beta, 1e24)
        end
    end # outer while loop

    return
end

function admm_multiperiod_restart_two_level(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    models::Array{ModelWithRamping{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},1}
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
                sol.z_prev .= sol.z_curr
                z_prev_norm += sum(sol.z_prev.^2)
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
                for g=1:inner.ngen
                    sol.xbar_curr[g] =
                       (sol_tp1.l_curr[3*g-2] + sol_tp1.rho[3*g-2]*(sol_tp1.x_curr[2*g-1] + sol_tp1.z_curr[3*g-2])
                        + sol.l_curr[3*g] + sol.rho[3*g]*(inner.solution.x_curr[2*g-1] + sol.z_curr[3*g])
                       ) / (sol_tp1.rho[3*g-2] + sol.rho[3*g])
                end
            end

            for t=2:env.horizon_length
                models[t].ramping_solution.xbar_tm1_curr .= models[t-1].ramping_solution.xbar_curr
            end

            # Update z for ramping.
            for t=2:env.horizon_length
                inner = models[t].inner
                sol = models[t].ramping_solution
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
            end

            for t=1:env.horizon_length-1
                inner = models[t].inner
                sol = models[t].ramping_solution
                for g=1:inner.ngen
                    # PG_TILDE
                    sol.z_curr[3*g] =
                        -(sol.lz[3*g] + sol.l_curr[3*g] + sol.rho[3*g]*(inner.solution.x_curr[2*g-1] - sol.xbar_curr[g])
                         ) / (beta + sol.rho[3*g])
                end
            end

            for t=1:env.horizon_length
                sol = models[t].ramping_solution
                sol.l_curr .= -(sol.lz .+ beta.*sol.z_curr)
            end

            for t=1:env.horizon_length
                inner = models[t].inner
                sol = models[t].ramping_solution
                for g=1:inner.ngen
                    if t > 1
                        sol.rp[3*g-2] = sol.x_curr[2*g-1] - sol.xbar_tm1_curr[g] + sol.z_curr[3*g-2]
                        sol.rp[3*g-1] = inner.solution.x_curr[2*g-1] - sol.x_curr[2*g-1] - sol.x_curr[2*g] + sol.z_curr[3*g-1]
                    end

                    if t < env.horizon_length
                        sol.rp[3*g] = inner.solution.x_curr[2*g-1] - sol.xbar_curr[g] + sol.z_curr[3*g]
                    end
                end
            end

            primres = 0.0
            dualres = 0.0
            mismatch = 0.0
            z_curr_norm = 0.0
            for t=1:env.horizon_length
                sol = models[t].ramping_solution
                primres += sum(sol.rp.^2)
                sol.Ax_plus_By .= sol.rp .- sol.z_curr
                mismatch += sum(sol.Ax_plus_By.^2)
                z_curr_norm += sum(sol.z_curr.^2)
                sol.rd .= sol.z_curr .- sol.z_prev
                dualres += sum(sol.rd.^2)
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

            #=
            for t=1:env.horizon_length
                @printf("t = %d\n", t)
                inner = models[t].inner
                sol = models[t].ramping_solution
                for g=1:inner.ngen
                    @printf("  PG = %.6e", inner.solution.x_curr[2*g-1])
                    @printf("  PG_BAR = %.6e", inner.solution.xbar_curr[2*g-1])
                    if t < env.horizon_length
                        @printf("  PG_TILDE = %.6e", sol.xbar_curr[g])
                        @printf("  Z_TILDE = %.6e", sol.z_curr[3*g])
                    end
                    if t > 1
                        @printf("  PG_HAT = %.6e", sol.x_curr[2*g-1])
                        @printf("  SG = %.6e", sol.x_curr[2*g])
                        @printf("  Z_SG = %.6e", sol.z_curr[3*g-1])
                    end
                    @printf("\n")
                end
            end
            =#
        end # while inner loop

        if mismatch <= OUTER_TOL
            break
        end

        for t=1:env.horizon_length
            sol = models[t].ramping_solution
            sol.lz .= max.(-par.MAX_MULTIPLIER,
                           min.(par.MAX_MULTIPLIER, sol.lz .+ (beta .* sol.z_curr)))
        end

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
function admm_multiperiod_two_level_cpu(
    case_prefix::String, load_prefix::String, horizon_length::Int;
    outer_iterlim::Int=20, inner_iterlim::Int=800,
    rho_pq::Float64=400.0, rho_va::Float64=40000.0, scale::Float64=1e-4,
    use_linelimit::Bool=false, verbose::Int=1
)
    env = AdmmEnv{Float64, Array{Float64,1}, Array{Int,1}, Array{Float64,2}}(
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