function get_generator_data(data::OPFData; use_gpu=false)
    ngen = length(data.generators)

    if use_gpu
        pgmin = CuArray{Float64}(undef, ngen)
        pgmax = CuArray{Float64}(undef, ngen)
        qgmin = CuArray{Float64}(undef, ngen)
        qgmax = CuArray{Float64}(undef, ngen)
        c2 = CuArray{Float64}(undef, ngen)
        c1 = CuArray{Float64}(undef, ngen)
        c0 = CuArray{Float64}(undef, ngen)
    else
        pgmin = Array{Float64}(undef, ngen)
        pgmax = Array{Float64}(undef, ngen)
        qgmin = Array{Float64}(undef, ngen)
        qgmax = Array{Float64}(undef, ngen)
        c2 = Array{Float64}(undef, ngen)
        c1 = Array{Float64}(undef, ngen)
        c0 = Array{Float64}(undef, ngen)
    end

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

function get_bus_data(data::OPFData; use_gpu=false)
    nbus = length(data.buses)

    FrIdx = [l for b=1:nbus for l in data.FromLines[b]]
    ToIdx = [l for b=1:nbus for l in data.ToLines[b]]
    GenIdx = [g for b=1:nbus for g in data.BusGenerators[b]]
    FrStart = accumulate(+, vcat([1], [length(data.FromLines[b]) for b=1:nbus]))
    ToStart = accumulate(+, vcat([1], [length(data.ToLines[b]) for b=1:nbus]))
    GenStart = accumulate(+, vcat([1], [length(data.BusGenerators[b]) for b=1:nbus]))

    Pd = Float64[data.buses[i].Pd for i=1:nbus]
    Qd = Float64[data.buses[i].Qd for i=1:nbus]
    Vmin = Float64[data.buses[i].Vmin for i=1:nbus]
    Vmax = Float64[data.buses[i].Vmax for i=1:nbus]

    if use_gpu
        cuFrIdx = CuArray{Int}(undef, length(FrIdx))
        cuToIdx = CuArray{Int}(undef, length(ToIdx))
        cuGenIdx = CuArray{Int}(undef, length(GenIdx))
        cuFrStart = CuArray{Int}(undef, length(FrStart))
        cuToStart = CuArray{Int}(undef, length(ToStart))
        cuGenStart = CuArray{Int}(undef, length(GenStart))
        cuPd = CuArray{Float64}(undef, nbus)
        cuQd = CuArray{Float64}(undef, nbus)
        cuVmax = CuArray{Float64}(undef, nbus)
        cuVmin = CuArray{Float64}(undef, nbus)

        copyto!(cuFrIdx, FrIdx)
        copyto!(cuToIdx, ToIdx)
        copyto!(cuGenIdx, GenIdx)
        copyto!(cuFrStart, FrStart)
        copyto!(cuToStart, ToStart)
        copyto!(cuGenStart, GenStart)
        copyto!(cuPd, Pd)
        copyto!(cuQd, Qd)
        copyto!(cuVmax, Vmax)
        copyto!(cuVmin, Vmin)

        return cuFrStart,cuFrIdx,cuToStart,cuToIdx,cuGenStart,cuGenIdx,cuPd,cuQd,cuVmin,cuVmax
    else
        return FrStart,FrIdx,ToStart,ToIdx,GenStart,GenIdx,Pd,Qd,Vmin,Vmax
    end
end

function get_branch_data(data::OPFData; use_gpu=false)
    buses = data.buses
    lines = data.lines
    BusIdx = data.BusIdx
    nline = length(data.lines)
    ybus = Ybus{Array{Float64}}(computeAdmitances(data.lines, data.buses, data.baseMVA; VI=Array{Int}, VD=Array{Float64})...)
    frBound = [ x for l=1:nline for x in (buses[BusIdx[lines[l].from]].Vmin^2, buses[BusIdx[lines[l].from]].Vmax^2) ]
    toBound = [ x for l=1:nline for x in (buses[BusIdx[lines[l].to]].Vmin^2, buses[BusIdx[lines[l].to]].Vmax^2) ]
    rateA = [ (data.lines[l].rateA / data.baseMVA)^2 for l=1:nline ]

    if use_gpu
        cuYshR = CuArray{Float64}(undef, length(ybus.YshR))
        cuYshI = CuArray{Float64}(undef, length(ybus.YshI))
        cuYffR = CuArray{Float64}(undef, nline)
        cuYffI = CuArray{Float64}(undef, nline)
        cuYftR = CuArray{Float64}(undef, nline)
        cuYftI = CuArray{Float64}(undef, nline)
        cuYttR = CuArray{Float64}(undef, nline)
        cuYttI = CuArray{Float64}(undef, nline)
        cuYtfR = CuArray{Float64}(undef, nline)
        cuYtfI = CuArray{Float64}(undef, nline)
        cuFrBound = CuArray{Float64}(undef, 2*nline)
        cuToBound = CuArray{Float64}(undef, 2*nline)
        cuRateA = CuArray{Float64}(undef, nline)
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
        copyto!(cuRateA, rateA)

        return cuYshR, cuYshI, cuYffR, cuYffI, cuYftR, cuYftI,
               cuYttR, cuYttI, cuYtfR, cuYtfI, cuFrBound, cuToBound, cuRateA
    else
        return ybus.YshR, ybus.YshI, ybus.YffR, ybus.YffI, ybus.YftR, ybus.YftI,
               ybus.YttR, ybus.YttI, ybus.YtfR, ybus.YtfI, frBound, toBound, rateA
    end
end

function init_solution!(model::Model{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    sol::SolutionOneLevel{Float64,Array{Float64,1}}, rho_pq::Float64, rho_va::Float64)

    ngen = model.ngen
    nline = model.nline

    brBusIdx = model.brBusIdx
    Vmax = model.Vmax; Vmin = model.Vmin
    YffR = model.YffR; YffI = model.YffI
    YttR = model.YttR; YttI = model.YttI
    YftR = model.YftR; YftI = model.YftI
    YtfR = model.YtfR; YtfI = model.YtfI

    for g=1:ngen
        pg_idx = model.gen_start + 2*(g-1)
        #u_curr[pg_idx] = 0.5*(data.genvec.Pmin[g] + data.genvec.Pmax[g])
        #u_curr[pg_idx+1] = 0.5*(data.genvec.Qmin[g] + data.genvec.Qmax[g])
        sol.v_curr[pg_idx] = 0.5*(model.pgmin[g] + model.pgmax[g])
        sol.v_curr[pg_idx+1] = 0.5*(model.qgmin[g] + model.qgmax[g])
    end

    sol.rho .= rho_pq
    fill!(sol.u_curr, 0.0)
    fill!(sol.v_curr, 0.0)

    for l=1:nline
        wij0 = (Vmax[brBusIdx[2*(l-1)+1]]^2 + Vmin[brBusIdx[2*(l-1)+1]]^2) / 2
        wji0 = (Vmax[brBusIdx[2*l]]^2 + Vmin[brBusIdx[2*l]]^2) / 2
        wR0 = sqrt(wij0 * wji0)

        pij_idx = model.line_start + 8*(l-1)
        sol.u_curr[pij_idx] = YffR[l] * wij0 + YftR[l] * wR0
        sol.u_curr[pij_idx+1] = -YffI[l] * wij0 - YftI[l] * wR0
        sol.u_curr[pij_idx+2] = YttR[l] * wji0 + YtfR[l] * wR0
        sol.u_curr[pij_idx+3] = -YttI[l] * wji0 - YtfI[l] * wR0
        #=
        u_curr[pij_idx+4] = wij0
        u_curr[pij_idx+5] = wji0
        u_curr[pij_idx+6] = 0.0
        u_curr[pij_idx+7] = 0.0
        =#
        # wRIij[2*(l-1)+1] = wR0
        # wRIij[2*l] = 0.0

        sol.v_curr[pij_idx+4] = wij0
        sol.v_curr[pij_idx+5] = wji0
        sol.v_curr[pij_idx+6] = 0.0
        sol.v_curr[pij_idx+7] = 0.0

        sol.rho[pij_idx+4:pij_idx+7] .= rho_va
    end

    sol.l_curr .= 0
    return
end

#=
function init_solution!(env::AdmmEnv, sol::SolutionOneLevel, ybus::Ybus, rho_pq, rho_va)
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
        pg_idx = model.gen_start + 2*(g-1)
        #u_curr[pg_idx] = 0.5*(data.genvec.Pmin[g] + data.genvec.Pmax[g])
        #u_curr[pg_idx+1] = 0.5*(data.genvec.Qmin[g] + data.genvec.Qmax[g])
        sol.v_curr[pg_idx] = 0.5*(data.generators[g].Pmin + data.generators[g].Pmax)
        sol.v_curr[pg_idx+1] = 0.5*(data.generators[g].Qmin + data.generators[g].Qmax)
    end

    sol.rho .= rho_pq
    fill!(sol.u_curr, 0.0)
    fill!(sol.v_curr, 0.0)

    for l=1:nline
        wij0 = (buses[BusIdx[lines[l].from]].Vmax^2 + buses[BusIdx[lines[l].from]].Vmin^2) / 2
        wji0 = (buses[BusIdx[lines[l].to]].Vmax^2 + buses[BusIdx[lines[l].to]].Vmin^2) / 2
        wR0 = sqrt(wij0 * wji0)

        pij_idx = model.line_start + 8*(l-1)
        sol.u_curr[pij_idx] = YffR[l] * wij0 + YftR[l] * wR0
        sol.u_curr[pij_idx+1] = -YffI[l] * wij0 - YftI[l] * wR0
        sol.u_curr[pij_idx+2] = YttR[l] * wji0 + YtfR[l] * wR0
        sol.u_curr[pij_idx+3] = -YttI[l] * wji0 - YtfI[l] * wR0
        #=
        u_curr[pij_idx+4] = wij0
        u_curr[pij_idx+5] = wji0
        u_curr[pij_idx+6] = 0.0
        u_curr[pij_idx+7] = 0.0
        =#
        # wRIij[2*(l-1)+1] = wR0
        # wRIij[2*l] = 0.0

        sol.v_curr[pij_idx+4] = wij0
        sol.v_curr[pij_idx+5] = wji0
        sol.v_curr[pij_idx+6] = 0.0
        sol.v_curr[pij_idx+7] = 0.0

        sol.rho[pij_idx+4:pij_idx+7] .= rho_va
    end

    sol.l_curr .= 0
    return
end
=#

function copy_data_kernel(n::Int, dest::CuDeviceArray{Float64,1}, src::CuDeviceArray{Float64,1})
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if tx <= n
        dest[tx] = src[tx]
    end
    return
end

function update_multiplier_kernel(n::Int, l_curr::CuDeviceArray{Float64,1},
                                  u_curr::CuDeviceArray{Float64,1},
                                  v_curr::CuDeviceArray{Float64,1},
                                  rho::CuDeviceArray{Float64,1})
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if tx <= n
        l_curr[tx] += rho[tx] * (u_curr[tx] - v_curr[tx])
    end
    return
end

function primal_residual_kernel(n::Int, rp::CuDeviceArray{Float64,1},
                                u_curr::CuDeviceArray{Float64,1},
                                v_curr::CuDeviceArray{Float64,1})
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if tx <= n
        rp[tx] = u_curr[tx] - v_curr[tx]
    end

    return
end

function dual_residual_kernel(n::Int, rd::CuDeviceArray{Float64,1},
                              v_prev::CuDeviceArray{Float64,1},
                              v_curr::CuDeviceArray{Float64,1},
                              rho::CuDeviceArray{Float64,1})
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if tx <= n
        rd[tx] = -rho[tx] * (v_curr[tx] - v_prev[tx])
    end

    return
end

function check_linelimit_violation(data::OPFData, u)
    lines = data.lines
    nline = length(data.lines)
    line_start = 2*length(data.generators) + 1

    rateA_nviols = 0
    rateA_maxviol = 0.0
    rateC_nviols = 0
    rateC_maxviol = 0.0

    for l=1:nline
        pij_idx = line_start + 8*(l-1)
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
    admm_restart

This function restarts the ADMM with a given `env::AdmmEnv` containing solutions and all the other parameters.
"""
function admm_restart(env::AdmmEnv, mod::Model; iterlim=800, scale=1e-4)
    if env.use_gpu
        CUDA.device!(env.gpu_no)
    end

    data, par, = env.data, env.params
    sol = mod.solution

    shift_lines = 0
    shmem_size = sizeof(Float64)*(14*mod.n+3*mod.n^2) + sizeof(Int)*(4*mod.n)

    nblk_gen = div(mod.ngen, 32, RoundUp)
    nblk_br = mod.nline
    nblk_bus = div(mod.nbus, 32, RoundUp)

    it = 0
    time_gen = time_br = time_bus = 0.0

    @time begin
    while it < iterlim
        it += 1

        if !env.use_gpu
            sol.u_prev .= sol.u_curr
            sol.v_prev .= sol.v_curr
            sol.l_prev .= sol.l_curr

            tcpu = generator_kernel(mod, mod.baseMVA, sol.u_curr, sol.v_curr, sol.l_curr, sol.rho)
            time_gen += tcpu.time

#            if env.use_polar
                tcpu = @timed auglag_it, tron_it = polar_kernel_cpu(mod.n, mod.nline, mod.line_start, scale,
                                                                    sol.u_curr, sol.v_curr, sol.l_curr, sol.rho,
                                                                    shift_lines, mod.membuf, mod.YffR, mod.YffI, mod.YftR, mod.YftI,
                                                                    mod.YttR, mod.YttI, mod.YtfR, mod.YtfI, mod.FrBound, mod.ToBound)
#            else
#                tcpu = @timed auglag_it, tron_it = auglag_kernel_cpu(mod.n, mod.nline, it, par.max_auglag, mod.line_start, par.mu_max,
#                                                                     sol.u_curr, sol.v_curr, sol.l_curr, sol.rho,
#                                                                     shift_lines, mod.membuf, mod.YffR, mod.YffI, mod.YftR, mod.YftI,
#                                                                     mod.YttR, mod.YttI, mod.YtfR, mod.YtfI, mod.FrBound, mod.ToBound)
#            end
            time_br += tcpu.time

            tcpu = @timed bus_kernel_cpu(mod.baseMVA, mod.nbus, mod.gen_start, mod.line_start,
                                         mod.FrStart, mod.FrIdx, mod.ToStart, mod.ToIdx, mod.GenStart,
                                         mod.GenIdx, mod.Pd, mod.Qd, sol.u_curr, sol.v_curr, sol.l_curr, sol.rho, mod.YshR, mod.YshI)
            time_bus += tcpu.time

            sol.l_curr .+= sol.rho .* (sol.u_curr .- sol.v_curr)
            sol.rd .= -sol.rho .* (sol.v_curr .- sol.v_prev)
            sol.rp .= sol.u_curr .- sol.v_curr
            #sol.rp_old .= sol.u_prev .- sol.v_prev

            primres = norm(sol.rp)
            dualres = norm(sol.rd)

            eps_pri = sqrt(length(sol.l_curr))*par.ABSTOL + par.RELTOL*max(norm(sol.u_curr), norm(-sol.v_curr))
            eps_dual = sqrt(length(sol.u_curr))*par.ABSTOL + par.RELTOL*norm(sol.l_curr)

            (par.verbose > 0) && @printf("[CPU] %10d  %.6e  %.6e  %.6e  %.6e  %6.2f  %6.2f\n",
                    it, primres, dualres, eps_pri, eps_dual, auglag_it, tron_it)

            if primres <= eps_pri && dualres <= eps_dual
                break
            end
        else
            @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) copy_data_kernel(mod.nvar, sol.u_prev, sol.u_curr)
            @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) copy_data_kernel(mod.nvar, sol.v_prev, sol.v_curr)
            @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) copy_data_kernel(mod.nvar, sol.l_prev, sol.l_curr)
            CUDA.synchronize()

            tgpu = generator_kernel(mod, mod.baseMVA, sol.u_curr, sol.v_curr, sol.l_curr, sol.rho)

            time_gen += tgpu.time
            if env.use_polar
                tgpu = CUDA.@timed @cuda threads=32 blocks=nblk_br shmem=shmem_size polar_kernel(mod.n, mod.nline, mod.line_start, scale,
                                                                                                 sol.u_curr, sol.v_curr, sol.l_curr, sol.rho,
                                                                                                 shift_lines, mod.membuf, mod.YffR, mod.YffI, mod.YftR, mod.YftI,
                                                                                                 mod.YttR, mod.YttI, mod.YtfR, mod.YtfI, mod.FrBound, mod.ToBound)
            else
                tgpu = CUDA.@timed @cuda threads=32 blocks=nblk_br shmem=shmem_size auglag_kernel(mod.n, it, par.max_auglag, mod.line_start, scale, par.mu_max,
                                                                                                  sol.u_curr, sol.v_curr, sol.l_curr, sol.rho,
                                                                                                  shift_lines, mod.membuf, mod.YffR, mod.YffI, mod.YftR, mod.YftI,
                                                                                                  mod.YttR, mod.YttI, mod.YtfR, mod.YtfI, mod.FrBound, mod.ToBound)
            end
            time_br += tgpu.time
            tgpu = CUDA.@timed @cuda threads=32 blocks=nblk_bus bus_kernel(mod.baseMVA, mod.nbus, mod.gen_start, mod.line_start,
                                                                           mod.FrStart, mod.FrIdx, mod.ToStart, mod.ToIdx, mod.GenStart,
                                                                           mod.GenIdx, mod.Pd, mod.Qd, sol.u_curr, sol.v_curr, sol.l_curr,
                                                                           sol.rho, mod.YshR, mod.YshI)
            time_bus += tgpu.time

            @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) update_multiplier_kernel(mod.nvar, sol.l_curr, sol.u_curr, sol.v_curr, sol.rho)
            @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) primal_residual_kernel(mod.nvar, sol.rp, sol.u_curr, sol.v_curr)
            @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) dual_residual_kernel(mod.nvar, sol.rd, sol.v_prev, sol.v_curr, sol.rho)
            CUDA.synchronize()

            gpu_primres = CUDA.norm(sol.rp)
            gpu_dualres = CUDA.norm(sol.rd)

            gpu_eps_pri = sqrt(length(sol.l_curr))*par.ABSTOL + par.RELTOL*max(CUDA.norm(sol.u_curr), CUDA.norm(sol.v_curr))
            gpu_eps_dual = sqrt(length(sol.u_curr))*par.ABSTOL + par.RELTOL*CUDA.norm(sol.l_curr)

            (par.verbose > 0) && @printf("[GPU] %10d  %.6e  %.6e  %.6e  %.6e\n", it, gpu_primres, gpu_dualres, gpu_eps_pri, gpu_eps_dual)

            if gpu_primres <= gpu_eps_pri && gpu_dualres <= gpu_eps_dual
                break
            end
        end
    end
    end

    u_curr = zeros(mod.nvar)
    copyto!(u_curr, sol.u_curr)
    objval = sum(data.generators[g].coeff[data.generators[g].n-2]*(mod.baseMVA*u_curr[mod.gen_start+2*(g-1)])^2 +
                 data.generators[g].coeff[data.generators[g].n-1]*(mod.baseMVA*u_curr[mod.gen_start+2*(g-1)]) +
                 data.generators[g].coeff[data.generators[g].n]
                 for g in 1:mod.ngen)::Float64
    sol.objval = objval


    if par.verbose > 0
        rateA_nviols, rateA_maxviol, rateC_nviols, rateC_maxviol = check_linelimit_violation(data, u_curr)
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
    end
    return
end

function admm_rect_gpu(case::String; iterlim=800, rho_pq=400.0, rho_va=40000.0, scale=1e-4,
                       use_gpu=false, solve_pf=false, gpu_no=0, verbose=1)
    if use_gpu
        CUDA.device!(gpu_no)

        env = AdmmEnv{Float64, CuArray{Float64, 1}, CuArray{Int, 1}, CuArray{Float64, 2}}(
            case, rho_pq, rho_va; use_gpu=use_gpu, use_twolevel=false, solve_pf=solve_pf, gpu_no=gpu_no, verbose=verbose,
        )
        mod = Model{Float64, CuArray{Float64,1}, CuArray{Int,1}, CuArray{Float64,2}}(env)
    else
        env = AdmmEnv{Float64, Array{Float64, 1}, Array{Int, 1}, Array{Float64, 2}}(
            case, rho_pq, rho_va; use_gpu=use_gpu, use_twolevel=false, solve_pf=solve_pf, gpu_no=gpu_no, verbose=verbose,
        )
        mod = Model{Float64, Array{Float64,1}, Array{Int,1}, Array{Float64,2}}(env)
    end

    admm_restart(env, mod, iterlim=iterlim, scale=scale)
    return env, mod
end

# TODO: This needs revised to use AdmmEnv.
function admm_rect_gpu_mpi(
    case::String;
    iterlim=800, rho_pq=400.0, rho_va=40000.0, scale=1e-4, use_gpu=false, use_polar=true, gpu_no=1,
    comm::MPI.Comm=MPI.COMM_WORLD,
)
    data = opf_loaddata(case)

    ngen = length(data.generators)
    nline = length(data.lines)
    nbus = length(data.buses)
    nvar = 2*ngen + 8*nline
    if use_gpu
        @assert MPI.has_cuda()
    end

    # MPI settings
    root = 0
    is_master = MPI.Comm_rank(comm) == root
    n_processes = MPI.Comm_size(comm)
    nlines_local = div(nline, n_processes, RoundUp)
    nlines_padded = n_processes * nlines_local

    nvar_padded = 2*ngen + 8 * nlines_padded

    baseMVA = data.baseMVA
    n = (use_polar == true) ? 4 : 10
    mu_max = 1e8
    rho_max = 1e6
    rho_min_pq = 5.0
    rho_min_w = 5.0
    eps_rp = 1e-4
    eps_rp_min = 1e-5
    rt_inc = 2.0
    rt_dec = 2.0
    eta = 0.99
    Kf = 100
    Kf_mean = 10

    if use_gpu
        CUDA.device!(MPI.Comm_rank(comm) % CUDA.ndevices())
    end

    ybus = Ybus{Array{Float64}}(computeAdmitances(data.lines, data.buses, data.baseMVA; VI=Array{Int}, VD=Array{Float64})...)

    pgmin, pgmax, qgmin, qgmax, c2, c1, c0 = get_generator_data(data)
    YshR, YshI, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI, FrBound, ToBound = get_branch_data(data)
    FrStart, FrIdx, ToStart, ToIdx, GenStart, GenIdx, Pd, Qd = get_bus_data(data)

    cu_pgmin, cu_pgmax, cu_qgmin, cu_qgmax, cu_c2, cu_c1, cu_c0 = get_generator_data(data; use_gpu=use_gpu)
    cuYshR, cuYshI, cuYffR, cuYffI, cuYftR, cuYftI, cuYttR, cuYttI, cuYtfR, cuYtfI, cuFrBound, cuToBound = get_branch_data(data; use_gpu=use_gpu)
    cu_FrStart, cu_FrIdx, cu_ToStart, cu_ToIdx, cu_GenStart, cu_GenIdx, cu_Pd, cu_Qd = get_bus_data(data; use_gpu=use_gpu)

    gen_start = 1
    line_start = 2*ngen + 1

    # Allocations
    u_curr = zeros(nvar_padded)
    v_curr = zeros(nvar_padded)
    l_curr = zeros(nvar_padded)
    u_prev = zeros(nvar_padded)
    v_prev = zeros(nvar_padded)
    l_prev = zeros(nvar_padded)
    rho = zeros(nvar_padded)
    rd = zeros(nvar_padded)
    rp = zeros(nvar_padded)
    rp_old = zeros(nvar_padded)
    rp_k0 = zeros(nvar_padded)
    param = zeros(31, nlines_padded)
    wRIij = zeros(2*nline)

    init_values(data, ybus, gen_start, line_start,
                rho_pq, rho_va, u_curr, v_curr, l_curr, rho, wRIij)


    if use_gpu
        cu_u_curr = CuArray{Float64}(undef, nvar_padded)
        cu_v_curr = CuArray{Float64}(undef, nvar_padded)
        cu_l_curr = CuArray{Float64}(undef, nvar_padded)
        cu_u_prev = CuArray{Float64}(undef, nvar_padded)
        cu_v_prev = CuArray{Float64}(undef, nvar_padded)
        cu_l_prev = CuArray{Float64}(undef, nvar_padded)
        cu_rho = CuArray{Float64}(undef, nvar_padded)
        cu_rd = CuArray{Float64}(undef, nvar_padded)
        cu_rp = CuArray{Float64}(undef, nvar_padded)
        cu_rp_old = CuArray{Float64}(undef, nvar_padded)
        cu_rp_k0 = CuArray{Float64}(undef, nvar_padded)
        cuParam = CuArray{Float64}(undef, (31, nlines_padded))
        cuWRIij = CuArray{Float64}(undef, 2*nline)

        copyto!(cu_u_curr, u_curr)
        copyto!(cu_v_curr, v_curr)
        copyto!(cu_l_curr, l_curr)
        copyto!(cu_rho, rho)
        copyto!(cuParam, param)
        copyto!(cuWRIij, wRIij)
    end

    # MPI: Global info
    if use_gpu
        u_lines_root = @view cu_u_curr[line_start:end]
        l_lines_root = @view cu_l_curr[line_start:end]
        v_lines_root = @view cu_v_curr[line_start:end]
        rho_lines_root = cu_rho[line_start:end]
    else
        u_lines_root = @view u_curr[line_start:end]
        l_lines_root = @view l_curr[line_start:end]
        v_lines_root = @view v_curr[line_start:end]
        rho_lines_root = @view rho[line_start:end]
    end
    # MPI: Local info
    # We need only to transfer info about lines
    if use_gpu
        u_local = CUDA.zeros(Float64, 8 * nlines_local)
        v_local = CUDA.zeros(Float64, 8 * nlines_local)
        l_local = CUDA.zeros(Float64, 8 * nlines_local)
        rho_local = CUDA.zeros(Float64, 8 * nlines_local)
    else
        u_local = zeros(8 * nlines_local)
        v_local = zeros(8 * nlines_local)
        l_local = zeros(8 * nlines_local)
        rho_local = zeros(8 * nlines_local)
    end

    MPI.Scatter!(u_lines_root, u_local, root, comm)
    MPI.Scatter!(rho_lines_root, rho_local, root, comm)

    max_auglag = 50

    nblk_gen = div(ngen, 32, RoundUp)
    nblk_br = nline
    nblk_br_local = nlines_local
    nblk_bus = div(nbus, 32, RoundUp)

    ABSTOL = 1e-6
    RELTOL = 1e-5

    it = 0
    time_gen = time_br = time_bus = 0
    time_br_scatter = time_br_gather = 0

    h_u_curr = zeros(nvar)
    h_param = zeros(31, nline)
    h_wRIij = zeros(2*nline)

    shift_lines = MPI.Comm_rank(comm) * nlines_local
    # GPU settings
    shmem_size = sizeof(Float64)*(14*n+3*n^2) + sizeof(Int)*(4*n)

    while it < iterlim
        it += 1

        # CPU code
        if !use_gpu

            if is_master
                u_prev .= u_curr
                v_prev .= v_curr
                l_prev .= l_curr
                tcpu = @timed generator_kernel_cpu(baseMVA, ngen, gen_start, u_curr, v_curr, l_curr, rho,
                                                pgmin, pgmax, qgmin, qgmax, c2, c1)
                time_gen += tcpu.time
            end

            # MPI routines to be implemented:
            #  - Broadcast cu_v_curr and cu_l_curr to GPUs.
            #  - Collect cu_u_curr.
            #  - div(nblk_br / number of GPUs, RoundUp)
            # scatter / gather

            tcpu_mpi = @timed begin
                MPI.Scatter!(v_lines_root, v_local, root, comm)
                MPI.Scatter!(l_lines_root, l_local, root, comm)
            end
            time_br_scatter += tcpu_mpi.time

            nlines_actual = min(nlines_local, nline - shift_lines)
            tcpu = @timed auglag_it, tron_it = polar_kernel_cpu(n, nlines_actual, 1, scale,
                                                                u_local, v_local, l_local, rho_local,
                                                                shift_lines, param, YffR, YffI, YftR, YftI,
                                                                YttR, YttI, YtfR, YtfI, FrBound, ToBound)

            time_br += tcpu.time
            tcpu_mpi = @timed MPI.Gather!(u_local, u_lines_root, root, comm)
            time_br_gather += tcpu_mpi.time

            if is_master
                tcpu = @timed bus_kernel_cpu(baseMVA, nbus, gen_start, line_start,
                                            FrStart, FrIdx, ToStart, ToIdx, GenStart,
                                            GenIdx, Pd, Qd, u_curr, v_curr, l_curr, rho, YshR, YshI)
                time_bus += tcpu.time

                l_curr .+= rho .* (u_curr .- v_curr)
                rd .= -rho .* (v_curr .- v_prev)
                rp .= u_curr .- v_curr
                #rp_old .= u_prev .- v_prev

                primres = norm(rp)
                dualres = norm(rd)

                eps_pri = sqrt(length(l_curr))*ABSTOL + RELTOL*max(norm(u_curr), norm(-v_curr))
                eps_dual = sqrt(length(u_curr))*ABSTOL + RELTOL*norm(l_curr)

                @printf("[CPU] %10d  %.6e  %.6e  %.6e  %.6e  %6.2f  %6.2f\n",
                        it, primres, dualres, eps_pri, eps_dual, auglag_it, tron_it)
            end
        # GPU code
        else
            if is_master
                @cuda threads=64 blocks=(div(nvar-1, 64)+1) copy_data_kernel(nvar, cu_u_prev, cu_u_curr)
                @cuda threads=64 blocks=(div(nvar-1, 64)+1) copy_data_kernel(nvar, cu_v_prev, cu_v_curr)
                @cuda threads=64 blocks=(div(nvar-1, 64)+1) copy_data_kernel(nvar, cu_l_prev, cu_l_curr)
                CUDA.synchronize()

                tgpu = CUDA.@timed @cuda threads=32 blocks=nblk_gen generator_kernel(baseMVA, ngen, gen_start,
                                                                                    cu_u_curr, cu_v_curr, cu_l_curr, cu_rho,
                                                                                    cu_pgmin, cu_pgmax, cu_qgmin, cu_qgmax, cu_c2, cu_c1)

                time_gen += tgpu.time
            end

            #  - Broadcast cu_v_curr and cu_l_curr to GPUs.
            tgpu_mpi = @timed begin
                MPI.Scatter!(v_lines_root, v_local, root, comm)
                MPI.Scatter!(l_lines_root, l_local, root, comm)
            end
            time_br_scatter += tgpu_mpi.time

            #  - div(nblk_br / number of GPUs, RoundUp)
            nblk_br_actual = min(nblk_br_local, nline - shift_lines)
            tgpu = CUDA.@timed @cuda threads=32 blocks=nblk_br_actual shmem=shmem_size polar_kernel(n, nline, 1, scale,
                u_local, v_local, l_local, rho_local,
                shift_lines, cuParam, cuYffR, cuYffI, cuYftR, cuYftI,
                cuYttR, cuYttI, cuYtfR, cuYtfI, cuFrBound, cuToBound
            )
            time_br += tgpu.time

            #  - Collect cu_u_curr.
            tgpu_mpi = @timed MPI.Gather!(u_local, u_lines_root, root, comm)
            time_br_gather += tgpu_mpi.time

            if is_master
                tgpu = CUDA.@timed @cuda threads=32 blocks=nblk_bus bus_kernel(baseMVA, nbus, gen_start, line_start,
                                                                            cu_FrStart, cu_FrIdx, cu_ToStart, cu_ToIdx, cu_GenStart,
                                                                            cu_GenIdx, cu_Pd, cu_Qd, cu_u_curr, cu_v_curr, cu_l_curr,
                                                                            cu_rho, cuYshR, cuYshI)
                time_bus += tgpu.time
                @cuda threads=64 blocks=(div(nvar-1, 64)+1) update_multiplier_kernel(nvar, cu_l_curr, cu_u_curr, cu_v_curr, cu_rho)
                @cuda threads=64 blocks=(div(nvar-1, 64)+1) primal_residual_kernel(nvar, cu_rp, cu_u_curr, cu_v_curr)
                @cuda threads=64 blocks=(div(nvar-1, 64)+1) dual_residual_kernel(nvar, cu_rd, cu_v_prev, cu_v_curr, cu_rho)
                CUDA.synchronize()

                gpu_primres = CUDA.norm(cu_rp)
                gpu_dualres = CUDA.norm(cu_rd)

                gpu_eps_pri = sqrt(length(l_curr))*ABSTOL + RELTOL*max(CUDA.norm(cu_u_curr), CUDA.norm(cu_v_curr))
                gpu_eps_dual = sqrt(length(u_curr))*ABSTOL + RELTOL*CUDA.norm(cu_l_curr)

                @printf("[GPU] %10d  %.6e  %.6e  %.6e  %.6e\n", it, gpu_primres, gpu_dualres, gpu_eps_pri, gpu_eps_dual)
            end
        end
    end

    if use_gpu
        copyto!(u_curr, cu_u_curr)
    end

    rank = MPI.Comm_rank(comm)
    @printf(" ** Time\n")
    @printf("[%d] Generator = %.2f\n", rank, time_gen)
    @printf("[%d] Branch    = %.2f\n", rank, time_br)
    @printf("[%d] Bus       = %.2f\n", rank, time_bus)
    @printf("[%d] G+Br+Bus  = %.2f\n", rank, time_gen + time_br + time_bus)
    @printf("[%d] Scatter   = %.2f\n", rank, time_br_scatter)
    @printf("[%d] Gather    = %.2f\n", rank, time_br_gather)
    @printf("[%d] MPI(S+G)  = %.2f\n", rank, time_br_scatter + time_br_gather)

    if is_master
        objval = sum(data.generators[g].coeff[data.generators[g].n-2]*(baseMVA*u_curr[gen_start+2*(g-1)])^2 +
                    data.generators[g].coeff[data.generators[g].n-1]*(baseMVA*u_curr[gen_start+2*(g-1)]) +
                    data.generators[g].coeff[data.generators[g].n]
                    for g in 1:ngen)
        @printf("Objective value = %.6e\n", objval)
    end

    return
end
