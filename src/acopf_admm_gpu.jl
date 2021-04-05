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
        u_curr[pg_idx] = 0.5*(data.genvec.Pmin[g] + data.genvec.Pmax[g])
        u_curr[pg_idx+1] = 0.5*(data.genvec.Qmin[g] + data.genvec.Qmax[g])
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
        u_curr[pij_idx+4] = wij0
        u_curr[pij_idx+5] = wji0
        u_curr[pij_idx+6] = 0.0
        u_curr[pij_idx+7] = 0.0
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

function admm_rect_gpu(case; iterlim=800, rho_pq=400.0, rho_va=40000.0, scale=1e-5,
                       use_gpu=false, use_polar=true, gpu_no=1)
    data = opf_loaddata(case)

    ngen = length(data.generators)
    nline = length(data.lines)
    nbus = length(data.buses)
    nvar = 2*ngen + 8*nline

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
        CUDA.device!(gpu_no)
    end

    ybus = Ybus{Array{Float64}}(computeAdmitances(data.lines, data.buses, data.baseMVA; VI=Array{Int}, VD=Array{Float64})...)

    pgmin, pgmax, qgmin, qgmax, c2, c1, c0 = get_generator_data(data)
    YshR, YshI, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI, FrBound, ToBound = get_branch_data(data)
    FrStart, FrIdx, ToStart, ToIdx, GenStart, GenIdx, Pd, Qd = get_bus_data(data)

    cu_pgmin, cu_pgmax, cu_qgmin, cu_qgmax, cu_c2, cu_c1, cu_c0 = get_generator_data(data; use_gpu=true)
    cuYshR, cuYshI, cuYffR, cuYffI, cuYftR, cuYftI, cuYttR, cuYttI, cuYtfR, cuYtfI, cuFrBound, cuToBound = get_branch_data(data; use_gpu=true)
    cu_FrStart, cu_FrIdx, cu_ToStart, cu_ToIdx, cu_GenStart, cu_GenIdx, cu_Pd, cu_Qd = get_bus_data(data; use_gpu=true)

    gen_start = 1
    line_start = 2*ngen + 1

    u_curr = zeros(nvar)
    v_curr = zeros(nvar)
    l_curr = zeros(nvar)
    u_prev = zeros(nvar)
    v_prev = zeros(nvar)
    l_prev = zeros(nvar)
    rho = zeros(nvar)
    rd = zeros(nvar)
    rp = zeros(nvar)
    rp_old = zeros(nvar)
    rp_k0 = zeros(nvar)
    param = zeros(31, nline)
    wRIij = zeros(2*nline)

    init_values(data, ybus, gen_start, line_start,
                rho_pq, rho_va, u_curr, v_curr, l_curr, rho, wRIij)

    cu_u_curr = CuArray{Float64}(undef, nvar)
    cu_v_curr = CuArray{Float64}(undef, nvar)
    cu_l_curr = CuArray{Float64}(undef, nvar)
    cu_u_prev = CuArray{Float64}(undef, nvar)
    cu_v_prev = CuArray{Float64}(undef, nvar)
    cu_l_prev = CuArray{Float64}(undef, nvar)
    cu_rho = CuArray{Float64}(undef, nvar)
    cu_rd = CuArray{Float64}(undef, nvar)
    cu_rp = CuArray{Float64}(undef, nvar)
    cu_rp_old = CuArray{Float64}(undef, nvar)
    cu_rp_k0 = CuArray{Float64}(undef, nvar)
    cuParam = CuArray{Float64}(undef, (31, nline))
    cuWRIij = CuArray{Float64}(undef, 2*nline)

    copyto!(cu_u_curr, u_curr)
    copyto!(cu_v_curr, v_curr)
    copyto!(cu_l_curr, l_curr)
    copyto!(cu_rho, rho)
    copyto!(cuParam, param)
    copyto!(cuWRIij, wRIij)

    max_auglag = 50

    nblk_gen = div(ngen, 32, RoundUp)
    nblk_br = nline
    nblk_bus = div(nbus, 32, RoundUp)

    ABSTOL = 1e-6
    RELTOL = 1e-5

    it = 0
    time_gen = time_br = time_bus = 0

    h_u_curr = zeros(nvar)
    h_param = zeros(31, nline)
    h_wRIij = zeros(2*nline)

    shmem_size = sizeof(Float64)*(14*n+3*n^2) + sizeof(Int)*(4*n)

    while it < iterlim
        it += 1

        if !use_gpu
            u_prev .= u_curr
            v_prev .= v_curr
            l_prev .= l_curr

            tcpu = @timed generator_kernel_cpu(baseMVA, ngen, gen_start, u_curr, v_curr, l_curr, rho,
                                               pgmin, pgmax, qgmin, qgmax, c2, c1)
            time_gen += tcpu.time

            if use_polar
                tcpu = @timed auglag_it, tron_it = polar_kernel_cpu(n, nline, line_start,
                                                                    u_curr, v_curr, l_curr, rho,
                                                                    param, YffR, YffI, YftR, YftI,
                                                                    YttR, YttI, YtfR, YtfI, FrBound, ToBound)
            else
                tcpu = @timed auglag_it, tron_it = auglag_kernel_cpu(n, nline, it, max_auglag, line_start, mu_max,
                                                                     u_curr, v_curr, l_curr, rho,
                                                                     wRIij, param, YffR, YffI, YftR, YftI,
                                                                     YttR, YttI, YtfR, YtfI, FrBound, ToBound)
            end
            time_br += tcpu.time
            tcpu = @timed bus_kernel_cpu(baseMVA, nbus, gen_start, line_start,
                                         FrStart, FrIdx, ToStart, ToIdx, GenStart,
                                         GenIdx, Pd, Qd, u_curr, v_curr, l_curr, rho, YshR, YshI)
            time_bus += tcpu.time

            l_curr .+= rho .* (u_curr .- v_curr)
            rd .= -rho .* (v_curr .- v_prev)
            rp .= u_curr .- v_curr
            rp_old .= u_prev .- v_prev

            primres = norm(rp)
            dualres = norm(rd)

            eps_pri = sqrt(length(l_curr))*ABSTOL + RELTOL*max(norm(u_curr), norm(-v_curr))
            eps_dual = sqrt(length(u_curr))*ABSTOL + RELTOL*norm(l_curr)

            @printf("[CPU] %10d  %.6e  %.6e  %.6e  %.6e  %6.2f  %6.2f\n",
                    it, primres, dualres, eps_pri, eps_dual, auglag_it, tron_it)
        else
            @cuda threads=64 blocks=(div(nvar-1, 64)+1) copy_data_kernel(nvar, cu_u_prev, cu_u_curr)
            @cuda threads=64 blocks=(div(nvar-1, 64)+1) copy_data_kernel(nvar, cu_v_prev, cu_v_curr)
            @cuda threads=64 blocks=(div(nvar-1, 64)+1) copy_data_kernel(nvar, cu_l_prev, cu_l_curr)
            CUDA.synchronize()

            tgpu = CUDA.@timed @cuda threads=32 blocks=nblk_gen generator_kernel(baseMVA, ngen, gen_start,
                                                                                 cu_u_curr, cu_v_curr, cu_l_curr, cu_rho,
                                                                                 cu_pgmin, cu_pgmax, cu_qgmin, cu_qgmax, cu_c2, cu_c1)
            # MPI routines to be implemented:
            #  - Broadcast cu_v_curr and cu_l_curr to GPUs.
            #  - Collect cu_u_curr.
            #  - div(nblk_br / number of GPUs, RoundUp)

            time_gen += tgpu.time
            if use_polar
                tgpu = CUDA.@timed @cuda threads=32 blocks=nblk_br shmem=shmem_size polar_kernel(n, line_start, scale,
                                                                                                 cu_u_curr, cu_v_curr, cu_l_curr, cu_rho,
                                                                                                 cuParam, cuYffR, cuYffI, cuYftR, cuYftI,
                                                                                                 cuYttR, cuYttI, cuYtfR, cuYtfI, cuFrBound, cuToBound)
            else
                tgpu = CUDA.@timed @cuda threads=32 blocks=nblk_br shmem=shmem_size auglag_kernel(n, it, max_auglag, line_start, scale, mu_max,
                                                                                                  cu_u_curr, cu_v_curr, cu_l_curr, cu_rho,
                                                                                                  cuWRIij, cuParam, cuYffR, cuYffI, cuYftR, cuYftI,
                                                                                                  cuYttR, cuYttI, cuYtfR, cuYtfI, cuFrBound, cuToBound)
            end
            time_br += tgpu.time
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

    if use_gpu
        copyto!(u_curr, cu_u_curr)
    end
    @printf(" ** Time\n")
    @printf("Generator = %.2f\n", time_gen)
    @printf("Branch    = %.2f\n", time_br)
    @printf("Bus       = %.2f\n", time_bus)
    @printf("Total     = %.2f\n", time_gen + time_br + time_bus)

    objval = sum(data.generators[g].coeff[data.generators[g].n-2]*(baseMVA*u_curr[gen_start+2*(g-1)])^2 +
                 data.generators[g].coeff[data.generators[g].n-1]*(baseMVA*u_curr[gen_start+2*(g-1)]) +
                 data.generators[g].coeff[data.generators[g].n]
                 for g in 1:ngen)
    @printf("Objective value = %.6e\n", objval)

    return
end
