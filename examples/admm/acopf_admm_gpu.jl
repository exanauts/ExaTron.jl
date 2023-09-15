function get_generator_data(data::OPFData, unknown)
    error("Unknown device $typeof(unknown)")
end

function get_generator_data(data::OPFData, backend::KA.Backend)
    ngen = length(data.generators)

    pgmin = adapt(backend, zeros(ngen))
    pgmax = adapt(backend, zeros(ngen))
    qgmin = adapt(backend, zeros(ngen))
    qgmax = adapt(backend, zeros(ngen))
    c2 = adapt(backend, zeros(ngen))
    c1 = adapt(backend, zeros(ngen))
    c0 = adapt(backend, zeros(ngen))

    Pmin = Float64[data.generators[g].Pmin for g in 1:ngen]
    Pmax = Float64[data.generators[g].Pmax for g in 1:ngen]
    Qmin = Float64[data.generators[g].Qmin for g in 1:ngen]
    Qmax = Float64[data.generators[g].Qmax for g in 1:ngen]
    coeff0 = Float64[data.generators[g].coeff[3] for g in 1:ngen]
    coeff1 = Float64[data.generators[g].coeff[2] for g in 1:ngen]
    coeff2 = Float64[data.generators[g].coeff[1] for g in 1:ngen]
    copyto!(pgmin, Pmin)
    copyto!(pgmax, Pmax)
    copyto!(qgmin, Qmin)
    copyto!(qgmax, Qmax)
    copyto!(c0, coeff0)
    copyto!(c1, coeff1)
    copyto!(c2, coeff2)

    return pgmin,pgmax,qgmin,qgmax,c2,c1,c0
end

function get_bus_data(data::OPFData, unknown)
    error("Unknown device $typeof(unknown)")
end

function get_bus_data(data::OPFData, backend::KA.Backend)
    nbus = length(data.buses)

    FrIdx = [l for b=1:nbus for l in data.FromLines[b]]
    ToIdx = [l for b=1:nbus for l in data.ToLines[b]]
    GenIdx = [g for b=1:nbus for g in data.BusGenerators[b]]
    FrStart = accumulate(+, vcat([1], [length(data.FromLines[b]) for b=1:nbus]))
    ToStart = accumulate(+, vcat([1], [length(data.ToLines[b]) for b=1:nbus]))
    GenStart = accumulate(+, vcat([1], [length(data.BusGenerators[b]) for b=1:nbus]))

    Pd = Float64[data.buses[i].Pd for i=1:nbus]
    Qd = Float64[data.buses[i].Qd for i=1:nbus]

    cuFrIdx = adapt(backend, zeros(Int, length(FrIdx)))
    cuToIdx = adapt(backend, zeros(Int, length(ToIdx)))
    cuGenIdx = adapt(backend, zeros(Int, length(GenIdx)))
    cuFrStart = adapt(backend, zeros(Int, length(FrStart)))
    cuToStart = adapt(backend, zeros(Int, length(ToStart)))
    cuGenStart = adapt(backend, zeros(Int, length(GenStart)))
    cuPd = adapt(backend, zeros(nbus))
    cuQd = adapt(backend, zeros(nbus))

    copyto!(cuFrIdx, FrIdx)
    copyto!(cuToIdx, ToIdx)
    copyto!(cuGenIdx, GenIdx)
    copyto!(cuFrStart, FrStart)
    copyto!(cuToStart, ToStart)
    copyto!(cuGenStart, GenStart)
    copyto!(cuPd, Pd)
    copyto!(cuQd, Qd)

    return cuFrStart,cuFrIdx,cuToStart,cuToIdx,cuGenStart,cuGenIdx,cuPd,cuQd
end

function get_branch_data(data::OPFData, unknown)
    error("Unknown device $typeof(unknown)")
end

function get_branch_data(data::OPFData, device::KA.Backend)
    buses = data.buses
    lines = data.lines
    BusIdx = data.BusIdx
    nline = length(data.lines)
    ybus = Ybus{Array{Float64}}(computeAdmitances(data.lines, data.buses, data.baseMVA, device; VI=Array{Int}, VD=Array{Float64})...)
    frBound = [ x for l=1:nline for x in (buses[BusIdx[lines[l].from]].Vmin^2, buses[BusIdx[lines[l].from]].Vmax^2) ]
    toBound = [ x for l=1:nline for x in (buses[BusIdx[lines[l].to]].Vmin^2, buses[BusIdx[lines[l].to]].Vmax^2) ]

    cuYshR = adapt(device, zeros(length(ybus.YshR)))
    cuYshI = adapt(device, zeros(length(ybus.YshI)))
    cuYffR = adapt(device, zeros(nline))
    cuYffI = adapt(device, zeros(nline))
    cuYftR = adapt(device, zeros(nline))
    cuYftI = adapt(device, zeros(nline))
    cuYttR = adapt(device, zeros(nline))
    cuYttI = adapt(device, zeros(nline))
    cuYtfR = adapt(device, zeros(nline))
    cuYtfI = adapt(device, zeros(nline))
    cuFrBound = adapt(device, zeros(2*nline))
    cuToBound = adapt(device, zeros(2*nline))
    copyto!(cuYshR, ybus.YshR)
    copyto!(cuYshI, ybus.YshI)
    copyto!(cuYffR, ybus.YffR)
    copyto!(cuYffI, ybus.YffI)
    copyto!(cuYftR, ybus.YftR)
    copyto!(cuYftI, ybus.YftI)
    copyto!(cuYttR, ybus.YttR)
    copyto!(cuYttI, ybus.YttI)
    copyto!(cuYtfR, ybus.YtfR)
    copyto!(cuYtfI, ybus.YtfI)
    copyto!(cuFrBound, frBound)
    copyto!(cuToBound, toBound)

    return cuYshR, cuYshI, cuYffR, cuYffI, cuYftR, cuYftI,
            cuYttR, cuYttI, cuYtfR, cuYtfI, cuFrBound, cuToBound
end

function init_solution!(env::AdmmEnv, sol::SolutionOneLevel, ybus::Ybus, rho_pq, rho_va, device)
    data, model = env.data, env.model

    lines = data.lines
    buses = data.buses
    BusIdx = data.BusIdx
    ngen = length(data.generators)
    nline = length(data.lines)

    YffR = ybus.YffR; YffI = ybus.YffI
    YttR = ybus.YttR; YttI = ybus.YttI
    YftR = ybus.YftR; YftI = ybus.YftI
    YtfR = ybus.YtfR; YtfI = ybus.YtfI

    for g=1:ngen
        pg_idx = model.gen_mod.gen_start + 2*(g-1)
        sol.v_curr[pg_idx:pg_idx+1] .= adapt(device, [
            0.5*(data.generators[g].Pmin + data.generators[g].Pmax),
            0.5*(data.generators[g].Qmin + data.generators[g].Qmax),
        ])
    end

    sol.rho .= rho_pq
    fill!(sol.u_curr, 0.0)
    fill!(sol.v_curr, 0.0)

    for l=1:nline
        wij0 = (buses[BusIdx[lines[l].from]].Vmax^2 + buses[BusIdx[lines[l].from]].Vmin^2) / 2
        wji0 = (buses[BusIdx[lines[l].to]].Vmax^2 + buses[BusIdx[lines[l].to]].Vmin^2) / 2
        wR0 = sqrt(wij0 * wji0)

        pij_idx = model.line_start + 8*(l-1)

        sol.u_curr[pij_idx:pij_idx+3] .= adapt(device, [
            YffR[l] * wij0 + YftR[l] * wR0,
            -YffI[l] * wij0 - YftI[l] * wR0,
            YttR[l] * wji0 + YtfR[l] * wR0,
            -YttI[l] * wji0 - YtfI[l] * wR0,
        ])
        #=
        u_curr[pij_idx+4] = wij0
        u_curr[pij_idx+5] = wji0
        u_curr[pij_idx+6] = 0.0
        u_curr[pij_idx+7] = 0.0
        =#
        # wRIij[2*(l-1)+1] = wR0
        # wRIij[2*l] = 0.0
        sol.v_curr[pij_idx+4:pij_idx+7] .= adapt(device, [
            wij0,
            wji0,
            0.0,
            0.0,
        ])

        sol.rho[pij_idx+4:pij_idx+7] .= rho_va
    end

    sol.l_curr .= 0
    return
end

@kernel function copy_data_kernel(n, dest, @Const(src))
    I_ = @index(Group, Linear)
    J_ = @index(Local, Linear)
    tx = J_ + (@groupsize()[1] * (I_ - 1))

    if tx <= n
        dest[tx] = src[tx]
    end
end

@kernel function update_multiplier_kernel(n::Int, l_curr,
                                  u_curr,
                                  v_curr,
                                  rho)
    I_ = @index(Group, Linear)
    J_ = @index(Local, Linear)
    tx = J_ + (@groupsize()[1] * (I_ - 1))

    if tx <= n
        l_curr[tx] += rho[tx] * (u_curr[tx] - v_curr[tx])
    end
end

@kernel function primal_residual_kernel(n::Int, rp,
                                u_curr,
                                v_curr)
    I_ = @index(Group, Linear)
    J_ = @index(Local, Linear)
    tx = J_ + (@groupsize()[1] * (I_ - 1))

    if tx <= n
        rp[tx] = u_curr[tx] - v_curr[tx]
    end
end

@kernel function dual_residual_kernel(n::Int, rd,
                              v_prev,
                              v_curr,
                              rho)
    I_ = @index(Group, Linear)
    J_ = @index(Local, Linear)
    tx = J_ + (@groupsize()[1] * (I_ - 1))

    if tx <= n
        rd[tx] = -rho[tx] * (v_curr[tx] - v_prev[tx])
    end
end

function check_linelimit_violation(data::OPFData, u_; device=KA.CPU())
    lines = data.lines
    nline = length(data.lines)
    line_start = 2*length(data.generators) + 1

    rateA_nviols = 0
    rateA_maxviol = 0.0
    rateC_nviols = 0
    rateC_maxviol = 0.0
    u = u_ |> Array

    for l=1:nline
        pij_idx = line_start + 8*(l-1)
        ij_val = 0.0
        ji_val = 0.0
        ij_val = u[pij_idx]^2 + u[pij_idx+1]^2
        ji_val = u[pij_idx+2]^2 + u[pij_idx+3]^2

        limit = (lines[l].rateA / data.baseMVA)^2
        if limit > 0 && limit < 1e10
            if ij_val > limit || ji_val > limit
                rateA_nviols += 1
                rateA_maxviol = max(rateA_maxviol, max(ij_val - limit, ji_val - limit))
            end
        end

        limit = (lines[l].rateC / data.baseMVA)^2
        if limit > 0 && limit < 1e10
            if ij_val > limit || ji_val > limit
                rateC_nviols += 1
                rateC_maxviol = max(rateC_maxviol, max(ij_val - limit, ji_val - limit))
            end
        end
    end
    rateA_maxviol = sqrt(rateA_maxviol)
    rateC_maxviol = sqrt(rateC_maxviol)

    return rateA_nviols, rateA_maxviol, rateC_nviols, rateC_maxviol
end

"""
    admm_restart!

This function restarts the ADMM with a given `env::AdmmEnv` containing solutions and all the other parameters.
"""
admm_restart!(env::AdmmEnv; options...) = admm_solve!(env, env.solution; options...)

function admm_solve!(env::AdmmEnv, sol::SolutionOneLevel; iterlim=800, scale=1e-4)

    data, par, mod = env.data, env.params, env.model

    shift_lines = 0
    shmem_size = sizeof(Float64)*(14*mod.n+3*mod.n^2) + sizeof(Int)*(4*mod.n)

    nblk_gen = div(mod.gen_mod.ngen, 32, RoundUp)
    nblk_br = mod.nline
    nblk_bus = div(mod.nbus, 32, RoundUp)

    it = 0
    time_gen = time_br = time_bus = 0.0
    primres = 0.0
    dualres = 0.0
    eps_pri = 0.0
    eps_dual = 0.0
    auglag_it = 0
    tron_it = 0
    gpu_primres = 0.0
    gpu_dualres = 0.0
    gpu_eps_pri = 0.0
    gpu_eps_dual = 0.0

    @time begin
    while it < iterlim
        it += 1

        if isa(env.device, KA.CPU)
            sol.u_prev .= sol.u_curr
            sol.v_prev .= sol.v_curr
            sol.l_prev .= sol.l_curr

            tcpu = generator_kernel(mod.gen_mod, data.baseMVA, sol.u_curr, sol.v_curr, sol.l_curr, sol.rho, env.device)
            time_gen += tcpu.time

            tcpu = @timed auglag_it, tron_it = polar_kernel_cpu(mod.n, mod.nline, mod.line_start, scale,
                                                                sol.u_curr, sol.v_curr, sol.l_curr, sol.rho,
                                                                shift_lines, env.membuf, mod.YffR, mod.YffI, mod.YftR, mod.YftI,
                                                                mod.YttR, mod.YttI, mod.YtfR, mod.YtfI, mod.FrBound, mod.ToBound)
            time_br += tcpu.time

            tcpu = @timed bus_kernel_cpu(data.baseMVA, mod.nbus, mod.gen_mod.gen_start, mod.line_start,
                                         mod.FrStart, mod.FrIdx, mod.ToStart, mod.ToIdx, mod.GenStart,
                                         mod.GenIdx, mod.Pd, mod.Qd, sol.u_curr, sol.v_curr, sol.l_curr, sol.rho, mod.YshR, mod.YshI)
            time_bus += tcpu.time

            sol.l_curr .+= sol.rho .* (sol.u_curr .- sol.v_curr)
            sol.rd .= -sol.rho .* (sol.v_curr .- sol.v_prev)
            sol.rp .= sol.u_curr .- sol.v_curr
            #sol.rp_old .= sol.u_prev .- sol.v_prev

            primres = norm(sol.rp)
            dualres = norm(sol.rd)

            eps_pri = sqrt(length(sol.l_curr))*par.abstol + par.reltol*max(norm(sol.u_curr), norm(-sol.v_curr))
            eps_dual = sqrt(length(sol.u_curr))*par.abstol + par.reltol*norm(sol.l_curr)

            (par.verbose > 1) && @printf("[CPU] %10d  %.6e  %.6e  %.6e  %.6e  %6.2f  %6.2f\n",
                    it, primres, dualres, eps_pri, eps_dual, auglag_it, tron_it)

            if primres <= eps_pri && dualres <= eps_dual
                break
            end
        else
            copy_data_kernel(env.device, 64, mod.nvar)(mod.nvar, sol.u_prev, sol.u_curr)
            copy_data_kernel(env.device, 64, mod.nvar)(mod.nvar, sol.v_prev, sol.v_curr)
            copy_data_kernel(env.device, 64, mod.nvar)(mod.nvar, sol.l_prev, sol.l_curr)
            KA.synchronize(env.device)

            generator_kernel(mod.gen_mod, data.baseMVA, sol.u_curr, sol.v_curr, sol.l_curr, sol.rho, env.device)

            polar_kernel(env.device, 32, mod.nline*32)(
                mod.n, mod.nline, mod.line_start, scale,
                sol.u_curr, sol.v_curr, sol.l_curr, sol.rho,
                shift_lines, env.membuf, mod.YffR, mod.YffI, mod.YftR, mod.YftI,
                mod.YttR, mod.YttI, mod.YtfR, mod.YtfI, mod.FrBound, mod.ToBound
            )
            KA.synchronize(env.device)

            bus_kernel(env.device, 32, mod.nbus)(
                data.baseMVA, mod.nbus, mod.gen_mod.gen_start, mod.line_start,
                mod.FrStart, mod.FrIdx, mod.ToStart, mod.ToIdx, mod.GenStart,
                mod.GenIdx, mod.Pd, mod.Qd, sol.u_curr, sol.v_curr, sol.l_curr,
                sol.rho, mod.YshR, mod.YshI
            )
            KA.synchronize(env.device)

            update_multiplier_kernel(env.device, 32, mod.nvar)(
                mod.nvar, sol.l_curr,
                sol.u_curr, sol.v_curr, sol.rho
            )
            KA.synchronize(env.device)

            primal_residual_kernel(env.device, 32, mod.nvar)(
                mod.nvar, sol.rp, sol.u_curr, sol.v_curr
            )
            KA.synchronize(env.device)

            dual_residual_kernel(env.device, 32, mod.nvar)(
                mod.nvar, sol.rd, sol.v_prev, sol.v_curr, sol.rho
            )
            KA.synchronize(env.device)

            gpu_primres = norm(sol.rp)
            gpu_dualres = norm(sol.rd)

            gpu_eps_pri = sqrt(length(sol.l_curr))*par.abstol + par.reltol*max(norm(sol.u_curr), norm(sol.v_curr))
            gpu_eps_dual = sqrt(length(sol.u_curr))*par.abstol + par.reltol*norm(sol.l_curr)

            (par.verbose > 1) && @printf("[GPU] %10d  %.6e  %.6e  %.6e  %.6e\n", it, gpu_primres, gpu_dualres, gpu_eps_pri, gpu_eps_dual)
            if Base.mod(it, 1000) == 0
                (par.verbose > 0) && @printf("[GPU] %10d  %.6e  %.6e  %.6e  %.6e\n", it, gpu_primres, gpu_dualres, gpu_eps_pri, gpu_eps_dual)
            end

            if gpu_primres <= gpu_eps_pri && gpu_dualres <= gpu_eps_dual
                break
            end
        end
    end
    end

    u_curr = zeros(mod.nvar)
    copyto!(u_curr, sol.u_curr)
    objval = sum(data.generators[g].coeff[data.generators[g].n-2]*(data.baseMVA*u_curr[mod.gen_mod.gen_start+2*(g-1)])^2 +
                 data.generators[g].coeff[data.generators[g].n-1]*(data.baseMVA*u_curr[mod.gen_mod.gen_start+2*(g-1)]) +
                 data.generators[g].coeff[data.generators[g].n]
                 for g in 1:mod.gen_mod.ngen)::Float64
    sol.objval = objval

    if it < iterlim
        sol.status = HAS_CONVERGED
    else
        sol.status = MAXIMUM_ITERATIONS
    end

    if par.verbose > 0
        rateA_nviols, rateA_maxviol, rateC_nviols, rateC_maxviol = check_linelimit_violation(data, u_curr)
        if isa(env.device, KA.CPU)
            @printf("[CPU] %10d  %.6e  %.6e  %.6e  %.6e  %6.2f  %6.2f\n", it, primres, dualres, eps_pri, eps_dual, auglag_it, tron_it)
        else
            @printf("[GPU] %10d  %.6e  %.6e  %.6e  %.6e\n", it, gpu_primres, gpu_dualres, gpu_eps_pri, gpu_eps_dual)
        end
        status = sol.status == HAS_CONVERGED ? "converged" : "not converged"
        @printf(" ** Line limit violations\n")
        @printf("RateA number of violations = %d (%d)\n", rateA_nviols, mod.nline)
        @printf("RateA maximum violation    = %.2f\n", rateA_maxviol)
        @printf("RateC number of violations = %d (%d)\n", rateC_nviols, mod.nline)
        @printf("RateC maximum violation    = %.2f\n", rateC_maxviol)

        @printf(" ** Time\n")
        @printf("Generator = %.2f\n", time_gen)
        @printf("Branch    = %.2f\n", time_br)
        @printf("Bus       = %.2f\n", time_bus)
        @printf("Total     = %.2f\n", time_gen + time_br + time_bus)

        @printf("Objective value = %.6e\n", objval)
        println("Solver status: $status")
        println("Iterations: $it")
    end
    return
end

function admm_gpu(case::String; iterlim=800, rho_pq=400.0, rho_va=40000.0, scale=1e-4,
                       device=KA.CPU(), verbose=1)
    env = AdmmEnv(case, device, rho_pq, rho_va; verbose=verbose,)
    admm_restart!(env, iterlim=iterlim, scale=scale)
    return env
end
