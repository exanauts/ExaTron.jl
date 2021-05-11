function get_generator_data(data; use_gpu=false)
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

    copyto!(pgmin, data.genvec.Pmin)
    copyto!(pgmax, data.genvec.Pmax)
    copyto!(qgmin, data.genvec.Qmin)
    copyto!(qgmax, data.genvec.Qmax)
    copyto!(c0, data.genvec.coeff0)
    copyto!(c1, data.genvec.coeff1)
    copyto!(c2, data.genvec.coeff2)

    return pgmin,pgmax,qgmin,qgmax,c2,c1,c0
end

function get_bus_data(data; use_gpu=false)
    ngen = length(data.generators)
    nbus = length(data.buses)
    nline = length(data.lines)

    FrIdx = [l for b=1:nbus for l in data.FromLines[b]]
    ToIdx = [l for b=1:nbus for l in data.ToLines[b]]
    GenIdx = [g for b=1:nbus for g in data.BusGenerators[b]]
    FrStart = accumulate(+, vcat([1], [length(data.FromLines[b]) for b=1:nbus]))
    ToStart = accumulate(+, vcat([1], [length(data.ToLines[b]) for b=1:nbus]))
    GenStart = accumulate(+, vcat([1], [length(data.BusGenerators[b]) for b=1:nbus]))

    Pd = [data.buses[i].Pd for i=1:nbus]
    Qd = [data.buses[i].Qd for i=1:nbus]

    if use_gpu
        cuFrIdx = CuArray{Int}(undef, length(FrIdx))
        cuToIdx = CuArray{Int}(undef, length(ToIdx))
        cuGenIdx = CuArray{Int}(undef, length(GenIdx))
        cuFrStart = CuArray{Int}(undef, length(FrStart))
        cuToStart = CuArray{Int}(undef, length(ToStart))
        cuGenStart = CuArray{Int}(undef, length(GenStart))
        cuPd = CuArray{Float64}(undef, nbus)
        cuQd = CuArray{Float64}(undef, nbus)

        copyto!(cuFrIdx, FrIdx)
        copyto!(cuToIdx, ToIdx)
        copyto!(cuGenIdx, GenIdx)
        copyto!(cuFrStart, FrStart)
        copyto!(cuToStart, ToStart)
        copyto!(cuGenStart, GenStart)
        copyto!(cuPd, Pd)
        copyto!(cuQd, Qd)

        return cuFrStart,cuFrIdx,cuToStart,cuToIdx,cuGenStart,cuGenIdx,cuPd,cuQd
    else
        return FrStart,FrIdx,ToStart,ToIdx,GenStart,GenIdx,Pd,Qd
    end
end

function get_branch_data(data; use_gpu=false)
    buses = data.buses
    lines = data.lines
    BusIdx = data.BusIdx
    nline = length(data.lines)
    ybus = Ybus{Array{Float64}}(computeAdmitances(data.lines, data.buses, data.baseMVA; VI=Array{Int}, VD=Array{Float64})...)
    frBound = [ x for l=1:nline for x in (buses[BusIdx[lines[l].from]].Vmin^2, buses[BusIdx[lines[l].from]].Vmax^2) ]
    toBound = [ x for l=1:nline for x in (buses[BusIdx[lines[l].to]].Vmin^2, buses[BusIdx[lines[l].to]].Vmax^2) ]

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
    else
        return ybus.YshR, ybus.YshI, ybus.YffR, ybus.YffI, ybus.YftR, ybus.YftI,
               ybus.YttR, ybus.YttI, ybus.YtfR, ybus.YtfI, frBound, toBound
    end
end

function init_values(data, ybus, gen_start, line_start,
                     rho_pq, rho_va, u_curr, v_curr, l_curr, rho, wRIij)
    lines = data.lines
    buses = data.buses
    BusIdx = data.BusIdx
    ngen = length(data.generators)
    nline = length(data.lines)

    YffR = ybus.YffR; YffI = ybus.YffI
    YttR = ybus.YttR; YttI = ybus.YttI
    YftR = ybus.YftR; YftI = ybus.YftI
    YtfR = ybus.YtfR; YtfI = ybus.YtfI
    YshR = ybus.YshR; YshI = ybus.YshI

    for g=1:ngen
        pg_idx = gen_start + 2*(g-1)
        #u_curr[pg_idx] = 0.5*(data.genvec.Pmin[g] + data.genvec.Pmax[g])
        #u_curr[pg_idx+1] = 0.5*(data.genvec.Qmin[g] + data.genvec.Qmax[g])
        v_curr[pg_idx] = 0.5*(data.genvec.Pmin[g] + data.genvec.Pmax[g])
        v_curr[pg_idx+1] = 0.5*(data.genvec.Qmin[g] + data.genvec.Qmax[g])
    end

    rho .= rho_pq

    for l=1:nline
        wij0 = (buses[BusIdx[lines[l].from]].Vmax^2 + buses[BusIdx[lines[l].from]].Vmin^2) / 2
        wji0 = (buses[BusIdx[lines[l].to]].Vmax^2 + buses[BusIdx[lines[l].to]].Vmin^2) / 2
        wR0 = sqrt(wij0 * wji0)

        pij_idx = line_start + 8*(l-1)
        u_curr[pij_idx] = YffR[l] * wij0 + YftR[l] * wR0
        u_curr[pij_idx+1] = -YffI[l] * wij0 - YftI[l] * wR0
        u_curr[pij_idx+2] = YttR[l] * wji0 + YtfR[l] * wR0
        u_curr[pij_idx+3] = -YttI[l] * wji0 - YtfI[l] * wR0
        #=
        u_curr[pij_idx+4] = wij0
        u_curr[pij_idx+5] = wji0
        u_curr[pij_idx+6] = 0.0
        u_curr[pij_idx+7] = 0.0
        =#
        wRIij[2*(l-1)+1] = wR0
        wRIij[2*l] = 0.0

        v_curr[pij_idx+4] = wij0
        v_curr[pij_idx+5] = wji0
        v_curr[pij_idx+6] = 0.0
        v_curr[pij_idx+7] = 0.0

        rho[pij_idx+4:pij_idx+7] .= rho_va
    end

    l_curr .= 0
    return
end

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

function check_linelimit_violation(data, u::Array{Float64,1})
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
    AdmmEnv{T,TD,TI}

This structure carries everything required to run ADMM from a given solution.
"""
mutable struct AdmmEnv{T,TD,TI}
    case::String
    data::OPFData
    use_gpu::Bool
    use_polar::Bool
    gpu_no::Int

    n::Int
    mu_max::Float64
    rho_max::Float64
    rho_min_pq::Float64
    rho_min_w::Float64
    eps_rp::Float64
    eps_rp_min::Float64
    rt_inc::Float64
    rt_dec::Float64
    eta::Float64

    max_auglag::Int

    ABSTOL::Float64
    RELTOL::Float64

    ngen::Int
    nline::Int
    nbus::Int
    nvar::Int

    pgmin::TD
    pgmax::TD
    qgmin::TD
    qgmax::TD
    c2::TD
    c1::TD
    c0::TD
    YshR::TD
    YshI::TD
    YffR::TD
    YffI::TD
    YftR::TD
    YftI::TD
    YttR::TD
    YttI::TD
    YtfR::TD
    YtfI::TD
    FrBound::TD
    ToBound::TD
    FrStart::TI
    FrIdx::TI
    ToStart::TI
    ToIdx::TI
    GenStart::TI
    GenIdx::TI
    Pd::TD
    Qd::TD

    u_curr::TD
    v_curr::TD
    l_curr::TD
    u_prev::TD
    v_prev::TD
    l_prev::TD
    rho::TD
    rd::TD
    rp::TD
    membuf::TD # was param

    gen_start::Int
    line_start::Int

    function AdmmEnv{T,TD,TI}(case, rho_pq, rho_va; use_gpu=false, use_polar=false, gpu_no=1) where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}}
        d = new{T,TD,TI}()

        d.case = case
        d.data = opf_loaddata(d.case)
        d.use_gpu = use_gpu
        d.use_polar = use_polar
        d.gpu_no = gpu_no
        
        d.n = (use_polar == true) ? 4 : 10
        d.mu_max = 1e8
        d.rho_max = 1e6
        d.rho_min_pq = 5.0
        d.rho_min_w = 5.0
        d.eps_rp = 1e-4
        d.eps_rp_min = 1e-5
        d.rt_inc = 2.0
        d.rt_dec = 2.0
        d.eta = 0.99

        d.max_auglag = 50
    
        d.ABSTOL = 1e-6
        d.RELTOL = 1e-5

        d.ngen = length(d.data.generators)
        d.nline = length(d.data.lines)
        d.nbus = length(d.data.buses)
        d.nvar = 2*d.ngen + 8*d.nline

        d.pgmin, d.pgmax, d.qgmin, d.qgmax, d.c2, d.c1, d.c0 = get_generator_data(d.data; use_gpu=use_gpu)
        d.YshR, d.YshI, d.YffR, d.YffI, d.YftR, d.YftI, d.YttR, d.YttI, d.YtfR, d.YtfI, d.FrBound, d.ToBound = get_branch_data(d.data; use_gpu=use_gpu)
        d.FrStart, d.FrIdx, d.ToStart, d.ToIdx, d.GenStart, d.GenIdx, d.Pd, d.Qd = get_bus_data(d.data; use_gpu=use_gpu)

        ybus = Ybus{Array{Float64}}(computeAdmitances(d.data.lines, d.data.buses, d.data.baseMVA; VI=Array{Int}, VD=Array{Float64})...)

        u_curr = zeros(d.nvar)
        v_curr = zeros(d.nvar)
        l_curr = zeros(d.nvar)
        u_prev = zeros(d.nvar)
        v_prev = zeros(d.nvar)
        l_prev = zeros(d.nvar)
        rho = zeros(d.nvar)
        rd = zeros(d.nvar)
        rp = zeros(d.nvar)
        membuf = zeros(31, d.nline)
        wRIij = zeros(2*d.nline)

        d.gen_start = 1
        d.line_start = 2*d.ngen + 1
    
        init_values(d.data, ybus, d.gen_start, d.line_start,
                    rho_pq, rho_va, u_curr, v_curr, l_curr, rho, wRIij)

        d.u_curr = TD(undef, d.nvar)
        d.v_curr = TD(undef, d.nvar)
        d.l_curr = TD(undef, d.nvar)
        d.u_prev = TD(undef, d.nvar)
        d.v_prev = TD(undef, d.nvar)
        d.l_prev = TD(undef, d.nvar)
        d.rho = TD(undef, d.nvar)
        d.rd = TD(undef, d.nvar)
        d.rp = TD(undef, d.nvar)
        d.membuf = TD(undef, (31, d.nline))

        copyto!(d.u_curr, u_curr)
        copyto!(d.v_curr, v_curr)
        copyto!(d.l_curr, l_curr)
        copyto!(d.rho, rho)
        copyto!(d.membuf, membuf)

        return d
    end
end

function admm_restart(env::AdmmEnv; iterlim=800, scale=1e-4)
    if env.use_gpu
        CUDA.device!(env.gpu_no)
    end

    data = env.data

    shift_lines = 0
    shmem_size = sizeof(Float64)*(14*env.n+3*env.n^2) + sizeof(Int)*(4*env.n)

    nblk_gen = div(env.ngen, 32, RoundUp)
    nblk_br = env.nline
    nblk_bus = div(env.nbus, 32, RoundUp)

    it = 0
    time_gen = time_br = time_bus = 0

    @time begin
    while it < iterlim
        it += 1

        if !env.use_gpu
            env.u_prev .= env.u_curr
            env.v_prev .= env.v_curr
            env.l_prev .= env.l_curr

            tcpu = @timed generator_kernel_cpu(data.baseMVA, env.ngen, env.gen_start, env.u_curr, env.v_curr, env.l_curr, env.rho,
                                               env.pgmin, env.pgmax, env.qgmin, env.qgmax, env.c2, env.c1)
            time_gen += tcpu.time

            if env.use_polar
                tcpu = @timed auglag_it, tron_it = polar_kernel_cpu(env.n, env.nline, env.line_start, scale,
                                                                    env.u_curr, env.v_curr, env.l_curr, env.rho,
                                                                    shift_lines, env.membuf, env.YffR, env.YffI, env.YftR, env.YftI,
                                                                    env.YttR, env.YttI, env.YtfR, env.YtfI, env.FrBound, env.ToBound)
            else
                tcpu = @timed auglag_it, tron_it = auglag_kernel_cpu(env.n, env.nline, it, env.max_auglag, env.line_start, env.mu_max,
                                                                     env.u_curr, env.v_curr, env.l_curr, env.rho,
                                                                     shift_lines, env.membuf, env.YffR, env.YffI, env.YftR, env.YftI,
                                                                     env.YttR, env.YttI, env.YtfR, env.YtfI, env.FrBound, env.ToBound)
            end
            time_br += tcpu.time

            tcpu = @timed bus_kernel_cpu(data.baseMVA, env.nbus, env.gen_start, env.line_start,
                                         env.FrStart, env.FrIdx, env.ToStart, env.ToIdx, env.GenStart,
                                         env.GenIdx, env.Pd, env.Qd, env.u_curr, env.v_curr, env.l_curr, env.rho, env.YshR, env.YshI)
            time_bus += tcpu.time

            env.l_curr .+= env.rho .* (env.u_curr .- env.v_curr)
            env.rd .= -env.rho .* (env.v_curr .- env.v_prev)
            env.rp .= env.u_curr .- env.v_curr
            #env.rp_old .= env.u_prev .- env.v_prev

            primres = norm(env.rp)
            dualres = norm(env.rd)

            eps_pri = sqrt(length(env.l_curr))*env.ABSTOL + env.RELTOL*max(norm(env.u_curr), norm(-env.v_curr))
            eps_dual = sqrt(length(env.u_curr))*env.ABSTOL + env.RELTOL*norm(env.l_curr)

            @printf("[CPU] %10d  %.6e  %.6e  %.6e  %.6e  %6.2f  %6.2f\n",
                    it, primres, dualres, eps_pri, eps_dual, auglag_it, tron_it)

            if primres <= eps_pri && dualres <= eps_dual
                break
            end
        else
            @cuda threads=64 blocks=(div(env.nvar-1, 64)+1) copy_data_kernel(env.nvar, env.u_prev, env.u_curr)
            @cuda threads=64 blocks=(div(env.nvar-1, 64)+1) copy_data_kernel(env.nvar, env.v_prev, env.v_curr)
            @cuda threads=64 blocks=(div(env.nvar-1, 64)+1) copy_data_kernel(env.nvar, env.l_prev, env.l_curr)
            CUDA.synchronize()

            tgpu = CUDA.@timed @cuda threads=32 blocks=nblk_gen generator_kernel(data.baseMVA, env.ngen, env.gen_start,
                                                                                 env.u_curr, env.v_curr, env.l_curr, env.rho,
                                                                                 env.pgmin, env.pgmax, env.qgmin, env.qgmax, env.c2, env.c1)

            time_gen += tgpu.time
            if env.use_polar
                tgpu = CUDA.@timed @cuda threads=32 blocks=nblk_br shmem=shmem_size polar_kernel(env.n, env.nline, env.line_start, scale,
                                                                                                 env.u_curr, env.v_curr, env.l_curr, env.rho,
                                                                                                 shift_lines, env.membuf, env.YffR, env.YffI, env.YftR, env.YftI,
                                                                                                 env.YttR, env.YttI, env.YtfR, env.YtfI, env.FrBound, env.ToBound)
            else
                tgpu = CUDA.@timed @cuda threads=32 blocks=nblk_br shmem=shmem_size auglag_kernel(env.n, it, env.max_auglag, env.line_start, scale, env.mu_max,
                                                                                                  env.u_curr, env.v_curr, env.l_curr, env.rho,
                                                                                                  shift_lines, env.membuf, env.YffR, env.YffI, env.YftR, env.YftI,
                                                                                                  env.YttR, env.YttI, env.YtfR, env.YtfI, env.FrBound, env.ToBound)
            end
            time_br += tgpu.time
            tgpu = CUDA.@timed @cuda threads=32 blocks=nblk_bus bus_kernel(data.baseMVA, env.nbus, env.gen_start, env.line_start,
                                                                           env.FrStart, env.FrIdx, env.ToStart, env.ToIdx, env.GenStart,
                                                                           env.GenIdx, env.Pd, env.Qd, env.u_curr, env.v_curr, env.l_curr,
                                                                           env.rho, env.YshR, env.YshI)
            time_bus += tgpu.time

            @cuda threads=64 blocks=(div(env.nvar-1, 64)+1) update_multiplier_kernel(env.nvar, env.l_curr, env.u_curr, env.v_curr, env.rho)
            @cuda threads=64 blocks=(div(env.nvar-1, 64)+1) primal_residual_kernel(env.nvar, env.rp, env.u_curr, env.v_curr)
            @cuda threads=64 blocks=(div(env.nvar-1, 64)+1) dual_residual_kernel(env.nvar, env.rd, env.v_prev, env.v_curr, env.rho)
            CUDA.synchronize()

            gpu_primres = CUDA.norm(env.rp)
            gpu_dualres = CUDA.norm(env.rd)

            gpu_eps_pri = sqrt(length(env.l_curr))*env.ABSTOL + env.RELTOL*max(CUDA.norm(env.u_curr), CUDA.norm(env.v_curr))
            gpu_eps_dual = sqrt(length(env.u_curr))*env.ABSTOL + env.RELTOL*CUDA.norm(env.l_curr)

            @printf("[GPU] %10d  %.6e  %.6e  %.6e  %.6e\n", it, gpu_primres, gpu_dualres, gpu_eps_pri, gpu_eps_dual)

            if gpu_primres <= gpu_eps_pri && gpu_dualres <= gpu_eps_dual
                break
            end
        end
    end
    end

    u_curr = zeros(env.nvar)
    copyto!(u_curr, env.u_curr)

    rateA_nviols, rateA_maxviol, rateC_nviols, rateC_maxviol = check_linelimit_violation(data, u_curr)
    @printf(" ** Line limit violations\n")
    @printf("RateA number of violations = %d (%d)\n", rateA_nviols, env.nline)
    @printf("RateA maximum violation    = %.2f\n", rateA_maxviol)
    @printf("RateC number of violations = %d (%d)\n", rateC_nviols, env.nline)
    @printf("RateC maximum violation    = %.2f\n", rateC_maxviol)

    @printf(" ** Time\n")
    @printf("Generator = %.2f\n", time_gen)
    @printf("Branch    = %.2f\n", time_br)
    @printf("Bus       = %.2f\n", time_bus)
    @printf("Total     = %.2f\n", time_gen + time_br + time_bus)

    objval = sum(data.generators[g].coeff[data.generators[g].n-2]*(data.baseMVA*u_curr[env.gen_start+2*(g-1)])^2 +
                 data.generators[g].coeff[data.generators[g].n-1]*(data.baseMVA*u_curr[env.gen_start+2*(g-1)]) +
                 data.generators[g].coeff[data.generators[g].n]
                 for g in 1:env.ngen)
    @printf("Objective value = %.6e\n", objval)
end

function admm_rect_gpu(case; iterlim=800, rho_pq=400.0, rho_va=40000.0, scale=1e-4,
                       use_gpu=false, use_polar=true, gpu_no=1)
    TArray = ifelse(use_gpu, CuArray, Array)
    if use_gpu
        CUDA.device!(gpu_no)
    end
    env = AdmmEnv{Float64,TArray{Float64},TArray{Int}}(case, rho_pq, rho_va; use_gpu=use_gpu, use_polar=use_polar, gpu_no=gpu_no)
    admm_restart(env, iterlim=iterlim, scale=scale)
    return env
end

# TODO: This needs revised to use AdmmEnv.
function admm_rect_gpu_mpi(
    case;
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
