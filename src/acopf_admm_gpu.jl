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

function admm_rect_gpu(case; rho_pq=400.0, rho_va=40000.0)
    data = opf_loaddata(case)

    ngen = length(data.generators)
    nline = length(data.lines)
    nbus = length(data.buses)
    nvar = 2*ngen + 6*nline

    baseMVA = data.baseMVA
    n = 8
    mu_max = 1e8
    rho_max = 1e6
    rho_min_pq = 5.0
    rho_min_w = 5.0
    eps_rp = 1e-4
    eps_rp_min = 1e-5
    rt_inc = 2
    rt_dec = 2
    eta = 0.99
    Kf = 100
    Kf_mean = 10

    ybus = Ybus{Array{Float64}}(computeAdmitances(data.lines, data.buses, data.baseMVA; VI=Array{Int}, VD=Array{Float64})...)

    pg_start = 0
    qg_start = ngen
    pij_start = 2*ngen
    qij_start = 2*ngen + nline
    pji_start = 2*ngen + 2*nline
    qji_start = 2*ngen + 3*nline
    wi_i_ij_start = 2*ngen + 4*nline
    wi_j_ji_start = 2*ngen + 5*nline

    u_curr = Array{Float64}(undef, nvar)
    v_curr = Array{Float64}(undef, nvar)
    l_curr = Array{Float64}(undef, nvar)
    u_prev = Array{Float64}(undef, nvar)
    v_prev = Array{Float64}(undef, nvar)
    l_prev = Array{Float64}(undef, nvar)
    rho = Array{Float64}(undef, nvar)
    rd = Array{Float64}(undef, nvar)
    rp = Array{Float64}(undef, nvar)
    rp_old = Array{Float64}(undef, nvar)
    rp_k0 = Array{Float64}(undef, nvar)
    param = Array{Float64}(undef, (24, nline))
    wRIij = Array{Float64}(undef, 2*nline)

    fill!(u_curr, 1.0)
    fill!(v_curr, 1.0)
    fill!(l_curr, 1.0)
    fill!(rho, 1.0)
    fill!(param, 0.0)
    fill!(wRIij, 0.0)

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
    cuParam = CuArray{Float64}(undef, (24, nline))
    cuWRIij = CuArray{Float64}(undef, 2*nline)

    copyto!(cu_u_curr, u_curr)
    copyto!(cu_v_curr, v_curr)
    copyto!(cu_l_curr, l_curr)
    copyto!(cu_rho, rho)
    copyto!(cuParam, param)
    copyto!(cuWRIij, wRIij)

    pgmin, pgmax, qgmin, qgmax, c2, c1, c0 = get_generator_data(data)
    FrStart, FrIdx, ToStart, ToIdx, GenStart, GenIdx, Pd, Qd = get_bus_data(data)
    #=
    YshR, YshI, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI, FrBound, ToBound = get_branch_data(data)

    auglag_kernel_cpu(n, nline, 1, pij_start, qij_start, pji_start, qji_start,
                      wi_i_ij_start, wi_j_ji_start, mu_max,
                      u_curr, v_curr, l_curr, rho,
                      wRIij, param, YffR, YffI, YftR, YftI,
                      YttR, YttI, YtfR, YtfI, FrBound, ToBound)
    =#

    cuYshR, cuYshI, cuYffR, cuYffI, cuYftR, cuYftI, cuYttR, cuYttI, cuYtfR, cuYtfI, cuFrBound, cuToBound = get_branch_data(data; use_gpu=true)
    t = @timed CUDA.@sync @cuda threads=(n,n) blocks=nline shmem=(sizeof(Float64)*(14*n+3*n^2) + sizeof(Int)*(4*n)) auglag_kernel(n, 1, pij_start, qij_start, pji_start, qji_start,
                                                                                                                   wi_i_ij_start, wi_j_ji_start, mu_max,
                                                                                                                   cu_u_curr, cu_v_curr, cu_l_curr, cu_rho,
                                                                                                                   cuWRIij, cuParam, cuYffR, cuYffI, cuYftR, cuYftI,
                                                                                                                   cuYttR, cuYttI, cuYtfR, cuYtfI, cuFrBound, cuToBound)

    h_u_curr = zeros(nvar)
    h_param = zeros(24, nline)
    h_wRIij = zeros(2*nline)
    copyto!(h_u_curr, cu_u_curr)
    copyto!(h_param, cuParam)
    copyto!(h_wRIij, cuWRIij)

    println("Time (GPU)           = ", t.time)
    println("Number of buses      = ", nbus)
    println("Number of generators = ", ngen)
    println("Number of lines      = ", nline)

    println("norm(u_curr - cu_u_curr) = ", norm(u_curr .- h_u_curr))
    println("norm(param - cuParam) = ", norm(param .- h_param))
    println("norm(wRIij - cuWRIij) = ", norm(wRIij .- h_wRIij))
    println("findmax(inf(u_curr - cu_u_curr) = ", findmax((abs.(u_curr .- h_u_curr))))

    #=
    generator_kernel_cpu(baseMVA, ngen, pg_start, qg_start,
                     u_curr, v_curr, l_curr, rho, pgmin, pgmax, qgmin, qgmax, c2, c1, c0)
    bus_kernel_cpu(baseMVA, nbus, pg_start, qg_start, pij_start, qij_start,
                   pji_start, qji_start, wi_i_ij_start, wi_j_ji_start,
                   FrStart, FrIdx, ToStart, ToIdx, GenStart, GenIdx,
                   Pd, Qd, u_curr, v_curr, l_curr, rho, ybus.YshR, ybus.YshI)

    cuPgmin, cuPgmax, cuQgmin, cuQgmax, cuC2, cuC1, cuC0 = get_generator_data(data; use_gpu=true)
    cuFrStart, cuFrIdx, cuToStart, cuToIdx, cuGenStart, cuGenIdx = get_bus_data(data; use_gpu=true)

    nblk_gen = div(ngen, 32, RoundUp)
    nblk_bus = div(nbus, 32, RoundUp)

    CUDA.@sync @cuda threads=32 blocks=nblk_gen generator_kernel(baseMVA, ngen, pg_start, qg_start,
                                                cu_u_curr, cu_v_curr, cu_l_curr, cu_rho, cuPgmin, cuPgmax, cuQgmin, cuQgmax, cuC2, cuC1, cuC0)
    CUDA.@sync @cuda threads=32 blocks=nblk_bus bus_kernel(baseMVA, nbus, pg_start, qg_start,
                                                pij_start, qij_start, pji_start, qji_start, wi_i_ij_start, wi_j_ji_start,
                                                cuFrStart, cuFrIdx, cuToStart, cuToIdx, cuGenStart, cuGenIdx,
                                                cuPd, cuQd, cu_u_curr, cu_v_curr, cu_l_curr, cu_rho, cuYshR, cuYshI)
    =#

    return
end