
function init_solution!(env::AdmmEnv, sol::SolutionPowerFlow, ybus::Ybus, rho_pq, rho_va)
    data, model = env.data, env.model
    gen_start, line_start, bus_start = model.gen_mod.gen_start, model.line_start, model.bus_start

    lines = data.lines
    buses = data.buses
    BusIdx = data.BusIdx
    ngen = length(data.generators)
    nline = length(data.lines)
    nbus = length(data.buses)

    YffR = ybus.YffR; YffI = ybus.YffI
    YttR = ybus.YttR; YttI = ybus.YttI
    YftR = ybus.YftR; YftI = ybus.YftI
    YtfR = ybus.YtfR; YtfI = ybus.YtfI

    # Init
    fill!(sol.x_curr, 0.0)
    fill!(sol.xbar_curr, 0.0)
    fill!(sol.z_outer, 0.0)
    fill!(sol.z_curr, 0.0)
    fill!(sol.z_prev, 0.0)
    fill!(sol.l_curr, 0.0)
    fill!(sol.lz, 0.0)
    fill!(sol.rho, 0.0)
    fill!(sol.rp, 0.0)
    fill!(sol.rp_old, 0.0)
    fill!(sol.rd, 0.0)
    fill!(sol.Ax_plus_By, 0.0)

    u_curr = view(sol.x_curr, 1:model.nvar_u)
    v_curr = view(sol.x_curr, model.nvar_u+1:model.nvar)
    rho_u = view(sol.rho, 1:model.nvar_u)
    rho_v = view(sol.rho, model.nvar_u+1:model.nvar)

    rho_u .= rho_pq
    rho_v .= rho_pq
    rho_v[bus_start:end] .= rho_va

    for g=1:ngen
        pg_idx = gen_start + 2*(g-1)
        sol.xbar_curr[pg_idx] = data.generators[g].Pg
        sol.xbar_curr[pg_idx+1] = data.generators[g].Qg
        rho_u[pg_idx] = 0.0
        rho_u[pg_idx+1] = 0.0
        rho_v[pg_idx] = 0.0
        rho_v[pg_idx+1] = 0.0
    end

    fill!(sol.wRIij, 0.0)
    for l=1:nline
        fr_idx = BusIdx[lines[l].from]
        to_idx = BusIdx[lines[l].to]

        wij0 = buses[fr_idx].Vm^2
        wji0 = buses[to_idx].Vm^2
        wR0 = sqrt(wij0 * wji0)

        u_pij_idx = line_start + 8*(l-1)
        v_pij_idx = line_start + 4*(l-1)
        v_curr[v_pij_idx] = u_curr[u_pij_idx] = YffR[l] * wij0 + YftR[l] * wR0
        v_curr[v_pij_idx+1] = u_curr[u_pij_idx+1] = -YffI[l] * wij0 - YftI[l] * wR0
        v_curr[v_pij_idx+2] = u_curr[u_pij_idx+2] = YttR[l] * wji0 + YtfR[l] * wR0
        v_curr[v_pij_idx+3] = u_curr[u_pij_idx+3] = -YttI[l] * wji0 - YtfI[l] * wR0

        rho_u[u_pij_idx+4:u_pij_idx+7] .= rho_va

        sol.wRIij[2*(l-1)+1] = wR0
        sol.wRIij[2*l] = 0.0
    end

    for b=1:nbus
        sol.xbar_curr[bus_start + 2*(b-1)] = buses[b].Vm
        sol.xbar_curr[bus_start + 2*(b-1)+1] = 0.0
        if model.bustype[b] != 1
            rho_v[bus_start + 2*(b-1)] = 0.0
        end
    end

    return
end

function pf_update_xbar(env::AdmmEnv, u, v, xbar, zu, zv, lu, lv, rho_u, rho_v)
    data = env.data
    ngen = length(data.generators)
    nline = length(data.lines)
    nbus = length(data.buses)

    model = env.model
    gen_start = model.gen_mod.gen_start
    line_start = model.line_start
    bus_start = model.bus_start
    FrStart = model.FrStart
    ToStart = model.ToStart
    FrIdx = model.FrIdx
    ToIdx = model.ToIdx

    ul_cur = line_start
    vl_cur = line_start
    for j=1:nline
        xbar[vl_cur:vl_cur+3] .= (lu[ul_cur:ul_cur+3] .+ rho_u[ul_cur:ul_cur+3].*(u[ul_cur:ul_cur+3] .+ zu[ul_cur:ul_cur+3]) .+
                                  lv[vl_cur:vl_cur+3] .+ rho_v[vl_cur:vl_cur+3].*(v[vl_cur:vl_cur+3] .+ zv[vl_cur:vl_cur+3])) ./
                                 (rho_u[ul_cur:ul_cur+3] .+ rho_v[vl_cur:vl_cur+3])
        ul_cur += 8
        vl_cur += 4
    end

    for b=1:nbus
        wi_sum = 0.0
        ti_sum = 0.0
        rho_wi_sum = 0.0
        rho_ti_sum = 0.0

        for j=FrStart[b]:FrStart[b+1]-1
            u_pij_idx = line_start + 8*(FrIdx[j]-1)
            wi_sum += lu[u_pij_idx+4] + rho_u[u_pij_idx+4]*(u[u_pij_idx+4] + zu[u_pij_idx+4])
            ti_sum += lu[u_pij_idx+6] + rho_u[u_pij_idx+6]*(u[u_pij_idx+6] + zu[u_pij_idx+6])
            rho_wi_sum += rho_u[u_pij_idx+4]
            rho_ti_sum += rho_u[u_pij_idx+6]
        end
        for j=ToStart[b]:ToStart[b+1]-1
            u_pij_idx = line_start + 8*(ToIdx[j]-1)
            wi_sum += lu[u_pij_idx+5] + rho_u[u_pij_idx+5]*(u[u_pij_idx+5] + zu[u_pij_idx+5])
            ti_sum += lu[u_pij_idx+7] + rho_u[u_pij_idx+7]*(u[u_pij_idx+7] + zu[u_pij_idx+7])
            rho_wi_sum += rho_u[u_pij_idx+5]
            rho_ti_sum += rho_u[u_pij_idx+7]
        end

        bus_cur = bus_start + 2*(b-1)
        wi_sum += lv[bus_cur] + rho_v[bus_cur]*(v[bus_cur] + zv[bus_cur])
        rho_wi_sum += rho_v[bus_cur]
        ti_sum += lv[bus_cur+1] + rho_v[bus_cur+1]*(v[bus_cur+1] + zv[bus_cur+1])
        rho_ti_sum += rho_v[bus_cur+1]
        xbar[bus_cur] = wi_sum / rho_wi_sum
        xbar[bus_cur+1] = ti_sum / rho_ti_sum
    end
end

function pf_update_zu(env::AdmmEnv, u, xbar, z, l, rho, lz, beta)
    data = env.data
    lines = data.lines
    BusIdx = data.BusIdx
    ngen = length(data.generators)
    nline = length(data.lines)

    model = env.model
    line_start, bus_start = model.line_start, model.bus_start

    ul_cur = line_start
    xl_cur = line_start
    for j=1:nline
        fr_idx = bus_start + 2*(BusIdx[lines[j].from]-1)
        to_idx = bus_start + 2*(BusIdx[lines[j].to]-1)

        z[ul_cur:ul_cur+3] .= (-(lz[ul_cur:ul_cur+3] .+ l[ul_cur:ul_cur+3] .+ rho[ul_cur:ul_cur+3].*(u[ul_cur:ul_cur+3] .- xbar[xl_cur:xl_cur+3]))) ./ (beta .+ rho[ul_cur:ul_cur+3])
        z[ul_cur+4] = (-(lz[ul_cur+4] + l[ul_cur+4] + rho[ul_cur+4]*(u[ul_cur+4] - xbar[fr_idx]))) / (beta + rho[ul_cur+4])
        z[ul_cur+5] = (-(lz[ul_cur+5] + l[ul_cur+5] + rho[ul_cur+5]*(u[ul_cur+5] - xbar[to_idx]))) / (beta + rho[ul_cur+5])
        z[ul_cur+6] = (-(lz[ul_cur+6] + l[ul_cur+6] + rho[ul_cur+6]*(u[ul_cur+6] - xbar[fr_idx+1]))) / (beta + rho[ul_cur+6])
        z[ul_cur+7] = (-(lz[ul_cur+7] + l[ul_cur+7] + rho[ul_cur+7]*(u[ul_cur+7] - xbar[to_idx+1]))) / (beta + rho[ul_cur+7])
        ul_cur += 8
        xl_cur += 4
    end
end

function pf_compute_primal_residual_u(env::AdmmEnv, rp_u, u, xbar, z)
    data = env.data
    lines = data.lines
    BusIdx = data.BusIdx
    ngen = length(data.generators)
    nline = length(data.lines)

    model = env.model
    line_start, bus_start = model.line_start, model.bus_start

    ul_cur = line_start
    xl_cur = line_start
    for j=1:nline
        fr_idx = bus_start + 2*(BusIdx[lines[j].from]-1)
        to_idx = bus_start + 2*(BusIdx[lines[j].to]-1)

        rp_u[ul_cur:ul_cur+3] .= u[ul_cur:ul_cur+3] .- xbar[xl_cur:xl_cur+3] .+ z[ul_cur:ul_cur+3]
        rp_u[ul_cur+4] = u[ul_cur+4] - xbar[fr_idx] + z[ul_cur+4]
        rp_u[ul_cur+5] = u[ul_cur+5] - xbar[to_idx] + z[ul_cur+5]
        rp_u[ul_cur+6] = u[ul_cur+6] - xbar[fr_idx+1] + z[ul_cur+6]
        rp_u[ul_cur+7] = u[ul_cur+7] - xbar[to_idx+1] + z[ul_cur+7]

        ul_cur += 8
        xl_cur += 4
    end
end

function admm_solve!(env::AdmmEnv, sol::SolutionPowerFlow; outer_iterlim=1, inner_iterlim=10, scale=1e-4)
    data, par, mod = env.data, env.params, env.model

    # -------------------------------------------------------------------
    # Variables are of two types: u and v
    #   u contains variables for generators and branches and
    #   v contains variables for buses.
    #
    # Variable layout:
    #   u: | (pg,qg)_g | (pij,qij,pji,qji,w_i_ij,w_j_ij,a_i_ij, a_i_ji)_ij |
    #   v: | (pg,qg)_g | (pij,qij,pji,qji)_ij | (w_i,theta_i)_i |
    #   xbar: same as v
    # -------------------------------------------------------------------

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
    wRIij = sol.wRIij

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

    nblk_gen = div(mod.gen_mod.ngen, 32, RoundUp)
    nblk_br = mod.nline
    nblk_bus = div(mod.nbus, 32, RoundUp)

    beta = 1e3
    c = 6.0
    theta = 0.8
    sqrt_d = sqrt(mod.nvar_u + mod.nvar_v)
    OUTER_TOL = sqrt_d*(par.outer_eps)

    outer = 0
    inner = 0

    time_gen = time_br = time_bus = 0
    shift_lines = 0
    shmem_size = sizeof(Float64)*(14*mod.n+3*mod.n^2) + sizeof(Int)*(4*mod.n)

    mismatch = Inf
    z_prev_norm = z_curr_norm = Inf

    # Return status
    status = INITIAL

    while outer < outer_iterlim
        outer += 1

        z_outer .= z_curr
        z_prev_norm = norm(z_outer)

        inner = 0
        while inner < inner_iterlim
            inner += 1

            z_prev .= z_curr
            rp_old .= rp

            tcpu = @timed auglag_it, tron_it = polar_kernel_two_level_cpu(mod.n, mod.nline, mod.line_start, mod.bus_start, scale,
                u_curr, xbar_curr, zu_curr, lu_curr, rho_u,
                shift_lines, env.membuf, mod.YffR, mod.YffI, mod.YftR, mod.YftI,
                mod.YttR, mod.YttI, mod.YtfR, mod.YtfI, mod.FrBound, mod.ToBound, mod.brBusIdx
            )
            time_br += tcpu.time


            tcpu = @timed bus_kernel_powerflow_cpu(data.baseMVA, mod.nbus, mod.gen_mod.gen_start, mod.line_start, mod.bus_start,
                                                   mod.FrStart, mod.FrIdx, mod.ToStart, mod.ToIdx, mod.GenStart, mod.GenIdx,
                                                   mod.Pd, mod.Qd, mod.bustype,
                                                   v_curr, xbar_curr, zv_curr, lv_curr, rho_v, mod.YshR, mod.YshI)
            time_bus += tcpu.time

            pf_update_xbar(env, u_curr, v_curr, xbar_curr, zu_curr, zv_curr, lu_curr, lv_curr, rho_u, rho_v)

            pf_update_zu(env, u_curr, xbar_curr, zu_curr, lu_curr, rho_u, lz_u, beta)
            zv_curr .= (-(lz_v .+ lv_curr .+ rho_v.*(v_curr .- xbar_curr))) ./ (beta .+ rho_v)

            l_curr .= -(lz .+ beta.*z_curr)

            pf_compute_primal_residual_u(env, rp_u, u_curr, xbar_curr, zu_curr)
            rp_v .= v_curr .- xbar_curr .+ zv_curr

            rd .= z_curr .- z_prev
            Ax_plus_By .= rp .- z_curr

            primres = norm(rp)
            dualres = norm(rd)
            z_curr_norm = norm(z_curr)
            mismatch = norm(Ax_plus_By)
            eps_pri = sqrt_d / (2500*outer)

            if par.verbose > 0
                if inner == 1 || (inner % 50) == 0
                    @printf("%8s  %8s  %10s  %10s  %10s  %10s  %10s  %10s\n",
                            "Outer", "Inner", "PrimRes", "EpsPrimRes", "DualRes", "||z||", "||Ax+By||", "Beta")
                end

                @printf("%8d  %8d  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e\n",
                        outer, inner, primres, eps_pri, dualres, z_curr_norm, mismatch, beta)
            end

            if primres <= eps_pri || dualres <= par.DUAL_TOL
                break
            end
        end

        if mismatch <= OUTER_TOL
            status = HAS_CONVERGED
            break
        end

        lz .+= max.(-par.MAX_MULTIPLIER, min.(par.MAX_MULTIPLIER, beta .* z_curr))

        if z_curr_norm > theta*z_prev_norm
            beta = min(c*beta, 1e24)
        end
    end

    # Move x onto host
    if outer >= outer_iterlim
        sol.status = MAXIMUM_ITERATIONS
    else
        sol.status = status
    end

    if par.verbose > 0
        @printf(" ** Time\n")
        @printf("Generator = %.2f\n", time_gen)
        @printf("Branch    = %.2f\n", time_br)
        @printf("Bus       = %.2f\n", time_bus)
        @printf("Total     = %.2f\n", time_gen + time_br + time_bus)
    end

    return
end
