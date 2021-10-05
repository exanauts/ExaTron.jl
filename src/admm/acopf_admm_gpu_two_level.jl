function check_generator_bounds(env::AdmmEnv, xbar)
    generators = env.data.generators
    gen_start = env.model.gen_mod.gen_start

    ngen = length(generators)

    if isa(env.device, CUDADevice)
        pgmin = CuArray{Float64}(undef, ngen)
        pgmax = CuArray{Float64}(undef, ngen)
        qgmin = CuArray{Float64}(undef, ngen)
        qgmax = CuArray{Float64}(undef, ngen)
    elseif isa(env.device, ROCDevice)
        pgmin = ROCArray{Float64}(undef, ngen)
        pgmax = ROCArray{Float64}(undef, ngen)
        qgmin = ROCArray{Float64}(undef, ngen)
        qgmax = ROCArray{Float64}(undef, ngen)
    else
        pgmin = Array{Float64}(undef, ngen)
        pgmax = Array{Float64}(undef, ngen)
        qgmin = Array{Float64}(undef, ngen)
        qgmax = Array{Float64}(undef, ngen)
    end

    Pmin = Float64[generators[g].Pmin for g in 1:ngen]
    Pmax = Float64[generators[g].Pmax for g in 1:ngen]
    Qmin = Float64[generators[g].Qmin for g in 1:ngen]
    Qmax = Float64[generators[g].Qmax for g in 1:ngen]
    copyto!(pgmin, Pmin)
    copyto!(pgmax, Pmax)
    copyto!(qgmin, Qmin)
    copyto!(qgmax, Qmax)

    max_viol_real = 0.0
    max_viol_reactive = 0.0

    for g=1:ngen
        pidx = gen_start + 2*(g-1)
        qidx = gen_start + 2*(g-1) + 1

        real_err = 0.0
        reactive_err = 0.0
        AMDGPU.@allowscalar real_err = max(max(0.0, xbar[pidx] - pgmax[g]), max(0.0, pgmin[g] - xbar[pidx]))
        AMDGPU.@allowscalar reactive_err = max(max(0.0, xbar[qidx] - qgmax[g]), max(0.0, qgmin[g] - xbar[qidx]))

        max_viol_real = (max_viol_real < real_err) ? real_err : max_viol_real
        max_viol_reactive = (max_viol_reactive < reactive_err) ? reactive_err : max_viol_reactive
    end

    return max_viol_real, max_viol_reactive
end

function check_voltage_bounds(env::AdmmEnv, xbar)

    buses = env.data.buses
    nbus = length(buses)
    bus_start = env.model.bus_start

    max_viol = 0.0

    for b=1:nbus
        bidx = bus_start + 2*(b-1)
        err = 0.0
        AMDGPU.@allowscalar err = max(max(0.0, xbar[bidx] - buses[b].Vmax^2), max(0.0, buses[b].Vmin^2 - xbar[bidx]))
        max_viol = (max_viol < err) ? err : max_viol
    end

    return max_viol
end

function check_power_balance_violation(env::AdmmEnv, xbar)
    data = env.data
    model = env.model
    gen_start, line_start, bus_start, YshR, YshI = model.gen_mod.gen_start, model.line_start, model.bus_start, model.YshR, model.YshI

    baseMVA = data.baseMVA
    nbus = length(data.buses)

    max_viol_real = 0.0
    max_viol_reactive = 0.0
    for b=1:nbus
        real_err = 0.0
        reactive_err = 0.0
        for g in data.BusGenerators[b]
            AMDGPU.@allowscalar real_err += xbar[gen_start + 2*(g-1)]
            AMDGPU.@allowscalar reactive_err += xbar[gen_start + 2*(g-1)+1]
        end

        real_err = 0.0
        reactive_err = 0.0
        AMDGPU.@allowscalar real_err -= (env.model.Pd[b] / baseMVA)
        AMDGPU.@allowscalar reactive_err -= (env.model.Qd[b] / baseMVA)

        #real_err -= (data.buses[b].Pd / baseMVA)
        #reactive_err -= (data.buses[b].Qd / baseMVA)

        for l in data.FromLines[b]
            AMDGPU.@allowscalar real_err -= xbar[line_start + 4*(l-1)]
            AMDGPU.@allowscalar reactive_err -= xbar[line_start + 4*(l-1) + 1]
        end

        for l in data.ToLines[b]
            AMDGPU.@allowscalar real_err -= xbar[line_start + 4*(l-1) + 2]
            AMDGPU.@allowscalar reactive_err -= xbar[line_start + 4*(l-1) + 3]
        end

        AMDGPU.@allowscalar real_err -= YshR[b] * xbar[bus_start + 2*(b-1)]
        AMDGPU.@allowscalar reactive_err += YshI[b] * xbar[bus_start + 2*(b-1)]

        max_viol_real = (max_viol_real < abs(real_err)) ? abs(real_err) : max_viol_real
        max_viol_reactive = (max_viol_reactive < abs(reactive_err)) ? abs(reactive_err) : max_viol_reactive
    end

    return max_viol_real, max_viol_reactive
end

function get_branch_bus_index(data::OPFData; device=KA.CPU())
    lines = data.lines
    BusIdx = data.BusIdx
    nline = length(lines)

    brBusIdx = [ x for l=1:nline for x in (BusIdx[lines[l].from], BusIdx[lines[l].to]) ]

    if isa(device, KA.GPU)
        if isa(device, CUDADevice)
            cu_brBusIdx = CuArray{Int}(undef, 2*nline)
        elseif isa(device, ROCDevice)
            cu_brBusIdx = ROCArray{Int}(undef, 2*nline)
        else
            error("Unknown device")
        end
        copyto!(cu_brBusIdx, brBusIdx)
        return cu_brBusIdx
    else
        return brBusIdx
    end
end

function init_solution!(env::AdmmEnv, sol::SolutionTwoLevel, ybus::Ybus, rho_pq, rho_va)
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
        AMDGPU.@allowscalar sol.xbar_curr[pg_idx] = 0.5*(data.generators[g].Pmin + data.generators[g].Pmax)
        AMDGPU.@allowscalar sol.xbar_curr[pg_idx+1] = 0.5*(data.generators[g].Qmin + data.generators[g].Qmax)
    end

    fill!(sol.wRIij, 0.0)
    for l=1:nline
        fr_idx = BusIdx[lines[l].from]
        to_idx = BusIdx[lines[l].to]

        wij0 = (buses[fr_idx].Vmax^2 + buses[fr_idx].Vmin^2) / 2
        wji0 = (buses[to_idx].Vmax^2 + buses[to_idx].Vmin^2) / 2
        wR0 = sqrt(wij0 * wji0)

        u_pij_idx = line_start + 8*(l-1)
        v_pij_idx = line_start + 4*(l-1)
        AMDGPU.@allowscalar v_curr[v_pij_idx] = u_curr[u_pij_idx] = YffR[l] * wij0 + YftR[l] * wR0
        AMDGPU.@allowscalar v_curr[v_pij_idx+1] = u_curr[u_pij_idx+1] = -YffI[l] * wij0 - YftI[l] * wR0
        AMDGPU.@allowscalar v_curr[v_pij_idx+2] = u_curr[u_pij_idx+2] = YttR[l] * wji0 + YtfR[l] * wR0
        AMDGPU.@allowscalar v_curr[v_pij_idx+3] = u_curr[u_pij_idx+3] = -YttI[l] * wji0 - YtfI[l] * wR0

        rho_u[u_pij_idx+4:u_pij_idx+7] .= rho_va

        AMDGPU.@allowscalar sol.wRIij[2*(l-1)+1] = wR0
        AMDGPU.@allowscalar sol.wRIij[2*l] = 0.0
    end

    for b=1:nbus
        AMDGPU.@allowscalar sol.xbar_curr[bus_start + 2*(b-1)] = (buses[b].Vmax^2 + buses[b].Vmin^2) / 2
        AMDGPU.@allowscalar sol.xbar_curr[bus_start + 2*(b-1)+1] = 0.0
    end

    return
end

function update_xbar(env::AdmmEnv, u, v, xbar, zu, zv, lu, lv, rho_u, rho_v)
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

    gen_end = gen_start + 2*ngen - 1
    xbar[gen_start:gen_end] .= (lu[gen_start:gen_end] .+ rho_u[gen_start:gen_end].*(u[gen_start:gen_end] .+ zu[gen_start:gen_end]) .+
                                lv[gen_start:gen_end] .+ rho_v[gen_start:gen_end].*(v[gen_start:gen_end] .+ zv[gen_start:gen_end])) ./
                               (rho_u[gen_start:gen_end] .+ rho_v[gen_start:gen_end])

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

@kernel function update_xbar_generator_kernel(n::Int, gen_start::Int, u, v, xbar, zu, zv, lu, lv, rho_u, rho_v)
    I = @index(Group, Linear)
    J = @index(Local, Linear)
    tx = J + (I * (I - 1))

    if tx <= n
        i = gen_start + 2*(tx - 1)
        @inbounds begin
            xbar[i] = (lu[i] + rho_u[i]*(u[i] + zu[i]) + lv[i] + rho_v[i]*(v[i] + zv[i])) / (rho_u[i] + rho_v[i])
            xbar[i+1] = (lu[i+1] + rho_u[i+1]*(u[i+1] + zu[i+1]) + lv[i+1] + rho_v[i+1]*(v[i+1] + zv[i+1])) / (rho_u[i+1] + rho_v[i+1])
        end
    end
end

@kernel function update_xbar_branch_kernel(n::Int, line_start::Int, u, v, xbar, zu, zv, lu, lv, rho_u, rho_v)
    I = @index(Group, Linear)
    J = @index(Local, Linear)
    tx = J + (I * (I - 1))

    if tx <= n
        ul_cur = line_start + 8*(tx-1)
        vl_cur = line_start + 4*(tx-1)
        @inbounds begin
            xbar[vl_cur] = (lu[ul_cur] + rho_u[ul_cur]*(u[ul_cur] + zu[ul_cur]) +
                            lv[vl_cur] + rho_v[vl_cur]*(v[vl_cur] + zv[vl_cur])) /
                           (rho_u[ul_cur] + rho_v[vl_cur])
            xbar[vl_cur+1] = (lu[ul_cur+1] + rho_u[ul_cur+1]*(u[ul_cur+1] + zu[ul_cur+1]) +
                              lv[vl_cur+1] + rho_v[vl_cur+1]*(v[vl_cur+1] + zv[vl_cur+1])) /
                             (rho_u[ul_cur+1] + rho_v[vl_cur+1])
            xbar[vl_cur+2] = (lu[ul_cur+2] + rho_u[ul_cur+2]*(u[ul_cur+2] + zu[ul_cur+2]) +
                             lv[vl_cur+2] + rho_v[vl_cur+2]*(v[vl_cur+2] + zv[vl_cur+2])) /
                            (rho_u[ul_cur+2] + rho_v[vl_cur+2])
            xbar[vl_cur+3] = (lu[ul_cur+3] + rho_u[ul_cur+3]*(u[ul_cur+3] + zu[ul_cur+3]) +
                              lv[vl_cur+3] + rho_v[vl_cur+3]*(v[vl_cur+3] + zv[vl_cur+3])) /
                             (rho_u[ul_cur+3] + rho_v[vl_cur+3])
        end
    end
end

@kernel function update_xbar_bus_kernel(n::Int, line_start::Int, bus_start::Int, FrStart, FrIdx, ToStart, ToIdx, u, v, xbar, zu, zv, lu, lv, rho_u, rho_v)
    I = @index(Group, Linear)
    J = @index(Local, Linear)
    b = J + (I * (I - 1))

    if b <= n
        wi_sum = 0.0
        ti_sum = 0.0
        rho_wi_sum = 0.0
        rho_ti_sum = 0.0

        @inbounds begin
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
end

# TODO: This function is not used.
function update_lu(data::OPFData, gen_start, line_start, bus_start, u, xbar, zu, l, rho)
    lines = data.lines
    BusIdx = data.BusIdx
    ngen = length(data.generators)
    nline = length(data.lines)

    gen_end = gen_start + 2*ngen - 1
    l[gen_start:gen_end] .+= rho[gen_start:gen_end] .* (u[gen_start:gen_end] .- xbar[gen_start:gen_end] .+ zu[gen_start:gen_end])

    ul_cur = line_start
    xl_cur = line_start
    for j=1:nline
        fr_idx = bus_start + 2*(BusIdx[lines[j].from]-1)
        to_idx = bus_start + 2*(BusIdx[lines[j].to]-1)

        l[ul_cur:ul_cur+3] .+= rho[ul_cur:ul_cur+3] .* (u[ul_cur:ul_cur+3] .- xbar[xl_cur:xl_cur+3] .+ zu[ul_cur:ul_cur+3])
        l[ul_cur+4] += rho[ul_cur+4] * (u[ul_cur+4] - xbar[fr_idx] + zu[ul_cur+4])
        l[ul_cur+5] += rho[ul_cur+5] * (u[ul_cur+5] - xbar[to_idx] + zu[ul_cur+5])
        l[ul_cur+6] += rho[ul_cur+6] * (u[ul_cur+6] - xbar[fr_idx+1] + zu[ul_cur+6])
        l[ul_cur+7] += rho[ul_cur+7] * (u[ul_cur+7] - xbar[to_idx+1] + zu[ul_cur+7])
        ul_cur += 8
        xl_cur += 4
    end
end

function update_zu(env::AdmmEnv, u, xbar, z, l, rho, lz, beta)
    data = env.data
    lines = data.lines
    BusIdx = data.BusIdx
    ngen = length(data.generators)
    nline = length(data.lines)

    model = env.model
    gen_start, line_start, bus_start = model.gen_mod.gen_start, model.line_start, model.bus_start

    gen_end = gen_start + 2*ngen - 1
    z[gen_start:gen_end] .= (-(lz[gen_start:gen_end] .+ l[gen_start:gen_end] .+ rho[gen_start:gen_end].*(u[gen_start:gen_end] .- xbar[gen_start:gen_end]))) ./ (beta .+ rho[gen_start:gen_end])

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

@kernel function update_zu_generator_kernel(n::Int, gen_start::Int, u, xbar, z, l, rho, lz, beta)
    I = @index(Group, Linear)
    J = @index(Local, Linear)
    tx = J + (I * (I - 1))

    if tx <= n
        i = gen_start  + 2*(tx - 1)
        @inbounds begin
            z[i] = (-(lz[i] + l[i] + rho[i]*(u[i] - xbar[i]))) / (beta + rho[i])
            z[i+1] = (-(lz[i+1] + l[i+1] + rho[i+1]*(u[i+1] - xbar[i+1]))) / (beta + rho[i+1])
        end
    end
end

@kernel function update_zu_branch_kernel(n::Int, line_start::Int, bus_start::Int, brBusIdx, u, xbar, z, l, rho, lz, beta)
    I = @index(Group, Linear)
    J = @index(Local, Linear)
    tx = J + (I * (I - 1))

    if tx <= n
        ul_cur = line_start + 8*(tx - 1)
        xl_cur = line_start + 4*(tx - 1)

        @inbounds begin
            fr_idx = bus_start + 2*(brBusIdx[2*(tx-1)+1] - 1)
            to_idx = bus_start + 2*(brBusIdx[2*tx] - 1)

            z[ul_cur] = (-(lz[ul_cur] + l[ul_cur] + rho[ul_cur]*(u[ul_cur] - xbar[xl_cur]))) / (beta + rho[ul_cur])
            z[ul_cur+1] = (-(lz[ul_cur+1] + l[ul_cur+1] + rho[ul_cur+1]*(u[ul_cur+1] - xbar[xl_cur+1]))) / (beta + rho[ul_cur+1])
            z[ul_cur+2] = (-(lz[ul_cur+2] + l[ul_cur+2] + rho[ul_cur+2]*(u[ul_cur+2] - xbar[xl_cur+2]))) / (beta + rho[ul_cur+2])
            z[ul_cur+3] = (-(lz[ul_cur+3] + l[ul_cur+3] + rho[ul_cur+3]*(u[ul_cur+3] - xbar[xl_cur+3]))) / (beta + rho[ul_cur+3])
            z[ul_cur+4] = (-(lz[ul_cur+4] + l[ul_cur+4] + rho[ul_cur+4]*(u[ul_cur+4] - xbar[fr_idx]))) / (beta + rho[ul_cur+4])
            z[ul_cur+5] = (-(lz[ul_cur+5] + l[ul_cur+5] + rho[ul_cur+5]*(u[ul_cur+5] - xbar[to_idx]))) / (beta + rho[ul_cur+5])
            z[ul_cur+6] = (-(lz[ul_cur+6] + l[ul_cur+6] + rho[ul_cur+6]*(u[ul_cur+6] - xbar[fr_idx+1]))) / (beta + rho[ul_cur+6])
            z[ul_cur+7] = (-(lz[ul_cur+7] + l[ul_cur+7] + rho[ul_cur+7]*(u[ul_cur+7] - xbar[to_idx+1]))) / (beta + rho[ul_cur+7])
        end
    end
end

@kernel function update_zv_kernel(n::Int, v, xbar, z, l, rho, lz, beta)
    I = @index(Group, Linear)
    J = @index(Local, Linear)
    tx = J + (I * (I - 1))

    if tx <= n
        @inbounds begin
            z[tx] = (-(lz[tx] + l[tx] + rho[tx]*(v[tx] - xbar[tx]))) / (beta + rho[tx])
        end
    end
end

@kernel function update_l_kernel(n::Int, l, z, lz, beta)
    I = @index(Group, Linear)
    J = @index(Local, Linear)
    tx = J + (I * (I - 1))

    if tx <= n
        @inbounds begin
            l[tx] = -(lz[tx] + beta*z[tx])
        end
    end
end

@kernel function compute_primal_residual_u(env::AdmmEnv, rp_u, u, xbar, z)
    data = env.data
    lines = data.lines
    BusIdx = data.BusIdx
    ngen = length(data.generators)
    nline = length(data.lines)

    model = env.model
    gen_start, line_start, bus_start = model.gen_mod.gen_start, model.line_start, model.bus_start

    gen_end = gen_start + 2*ngen - 1
    rp_u[gen_start:gen_end] .= u[gen_start:gen_end] .- xbar[gen_start:gen_end] .+ z[gen_start:gen_end]

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

@kernel function compute_primal_residual_u_generator_kernel(n::Int, gen_start::Int, rp, u, xbar, z)
    I = @index(Group, Linear)
    J = @index(Local, Linear)
    tx = J + (I * (I - 1))

    if tx <= n
        i = gen_start + 2*(tx - 1)
        @inbounds begin
            rp[i] = u[i] - xbar[i] + z[i]
            rp[i+1] = u[i+1] - xbar[i+1] + z[i+1]
        end
    end
end

@kernel function compute_primal_residual_u_branch_kernel(n::Int, line_start::Int, bus_start::Int, brBusIdx, rp, u, xbar, z)
    I = @index(Group, Linear)
    J = @index(Local, Linear)
    tx = J + (I * (I - 1))

    if tx <= n
        ul_cur = line_start + 8*(tx - 1)
        xl_cur = line_start + 4*(tx - 1)
        fr_idx = bus_start + 2*(brBusIdx[2*(tx-1)+1]-1)
        to_idx = bus_start + 2*(brBusIdx[2*tx]-1)

        rp[ul_cur] = u[ul_cur] - xbar[xl_cur] + z[ul_cur]
        rp[ul_cur+1] = u[ul_cur+1] - xbar[xl_cur+1] + z[ul_cur+1]
        rp[ul_cur+2] = u[ul_cur+2] - xbar[xl_cur+2] + z[ul_cur+2]
        rp[ul_cur+3] = u[ul_cur+3] - xbar[xl_cur+3] + z[ul_cur+3]
        rp[ul_cur+4] = u[ul_cur+4] - xbar[fr_idx] + z[ul_cur+4]
        rp[ul_cur+5] = u[ul_cur+5] - xbar[to_idx] + z[ul_cur+5]
        rp[ul_cur+6] = u[ul_cur+6] - xbar[fr_idx+1] + z[ul_cur+6]
        rp[ul_cur+7] = u[ul_cur+7] - xbar[to_idx+1] + z[ul_cur+7]
    end
end

@kernel function compute_primal_residual_v_kernel(n::Int, rp, v, xbar, z)
    I = @index(Group, Linear)
    J = @index(Local, Linear)
    tx = J + (I * (I - 1))

    if tx <= n
        rp[tx] = v[tx] - xbar[tx] + z[tx]
    end
end

@kernel function vector_difference(n::Int, c, a, b)
    I = @index(Group, Linear)
    J = @index(Local, Linear)
    tx = J + (I * (I - 1))

    if tx <= n
        c[tx] = a[tx] - b[tx]
    end
end

@kernel function update_lz_kernel(n::Int, max_limit::Float64, z, lz, beta)
    I = @index(Group, Linear)
    J = @index(Local, Linear)
    tx = J + (I * (I - 1))

    if tx <= n
        lz[tx] += max(-max_limit, min(max_limit, beta*z[tx]))
    end
end

# TODO: Not used
function update_rho(rho, rp, rp_old, theta, gamma)
    for i=1:length(rho)
        if abs(rp[i]) > theta*abs(rp_old[i])
            rho[i] = min(gamma*rho[i], 1e24)
        end
    end
end

function admm_solve!(env::AdmmEnv, sol::SolutionTwoLevel; outer_iterlim=10, inner_iterlim=800, scale=1e-4)
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
    gamma = 6.0 # TODO: not used
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

        if isa(env.device, KA.CPU)
            z_outer .= z_curr
            z_prev_norm = norm(z_outer)
        else
            # synchronize(env.device)
            wait(copy_data_kernel(env.device)(z_outer, z_curr, ndrange = mod.nvar, dependencies=Event(env.device)))
            z_prev_norm = norm(z_curr)
        end

        inner = 0
        while inner < inner_iterlim
            inner += 1

            # if !env.use_gpu
            #     z_prev .= z_curr
            #     rp_old .= rp

            #     tcpu = generator_kernel_two_level(mod.gen_mod, data.baseMVA, u_curr, xbar_curr, zu_curr, lu_curr, rho_u)
            #     time_gen += tcpu.time

            #     #scale = min(scale, (2*1e4) / maximum(abs.(rho_u)))
            #     if env.use_polar
            #         tcpu = @timed auglag_it, tron_it = polar_kernel_two_level_cpu(mod.n, mod.nline, mod.line_start, mod.bus_start, scale,
            #                                                                     u_curr, xbar_curr, zu_curr, lu_curr, rho_u,
            #                                                                     shift_lines, env.membuf, mod.YffR, mod.YffI, mod.YftR, mod.YftI,
            #                                                                     mod.YttR, mod.YttI, mod.YtfR, mod.YtfI, mod.FrBound, mod.ToBound, mod.brBusIdx)
            #     else
            #         tcpu = @timed auglag_it, tron_it = auglag_kernel_cpu(mod.n, mod.nline, inner, par.max_auglag, mod.line_start, par.mu_max,
            #                                                             u_curr, v_curr, l_curr, rho,
            #                                                             wRIij, env.membuf, mod.YffR, mod.YffI, mod.YftR, mod.YftI,
            #                                                             mod.YttR, mod.YttI, mod.YtfR, mod.YtfI, mod.FrBound, mod.ToBound)
            #     end
            #     time_br += tcpu.time

            #     if !env.allow_infeas
            #         tcpu = @timed bus_kernel_two_level_cpu(data.baseMVA, mod.nbus, mod.gen_mod.gen_start, mod.line_start, mod.bus_start,
            #                                             mod.FrStart, mod.FrIdx, mod.ToStart, mod.ToIdx, mod.GenStart, mod.GenIdx,
            #                                             mod.Pd, mod.Qd, v_curr, xbar_curr, zv_curr, lv_curr, rho_v, mod.YshR, mod.YshI)
            #     else
            #         tcpu = @timed bus_kernel_two_level_cpu(data.baseMVA, mod.nbus, mod.gen_mod.gen_start, mod.line_start, mod.bus_start,
            #                                             mod.FrStart, mod.FrIdx, mod.ToStart, mod.ToIdx, mod.GenStart, mod.GenIdx,
            #                                             mod.Pd, mod.Qd, v_curr, xbar_curr, zv_curr, lv_curr, rho_v, mod.YshR, mod.YshI,
            #                                             sol.s_curr, par.rho_sigma)

            #     end
            #     time_bus += tcpu.time

            #     update_xbar(env, u_curr, v_curr, xbar_curr, zu_curr, zv_curr, lu_curr, lv_curr, rho_u, rho_v)

            #     update_zu(env, u_curr, xbar_curr, zu_curr, lu_curr, rho_u, lz_u, beta)
            #     zv_curr .= (-(lz_v .+ lv_curr .+ rho_v.*(v_curr .- xbar_curr))) ./ (beta .+ rho_v)

            #     l_curr .= -(lz .+ beta.*z_curr)

            #     compute_primal_residual_u(env, rp_u, u_curr, xbar_curr, zu_curr)
            #     rp_v .= v_curr .- xbar_curr .+ zv_curr

            #     #=
            #     if inner > 1
            #         update_rho(rho, rp, rp_old, theta, gamma)
            #     end
            #     =#

            #     rd .= z_curr .- z_prev
            #     Ax_plus_By .= rp .- z_curr

            #     primres = norm(rp)
            #     dualres = norm(rd)
            #     z_curr_norm = norm(z_curr)
            #     mismatch = norm(Ax_plus_By)
            #     eps_pri = sqrt_d / (2500*outer)

            #     if par.verbose > 0
            #         if inner == 1 || (inner % 50) == 0
            #             @printf("%8s  %8s  %10s  %10s  %10s  %10s  %10s  %10s\n",
            #                     "Outer", "Inner", "PrimRes", "EpsPrimRes", "DualRes", "||z||", "||Ax+By||", "Beta")
            #         end

            #         @printf("%8d  %8d  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e\n",
            #                 outer, inner, primres, eps_pri, dualres, z_curr_norm, mismatch, beta)
            #     end

            #     if primres <= eps_pri || dualres <= par.DUAL_TOL
            #         break
            #     end
            # else
                # synchronize(env.device)
                wait(copy_data_kernel(env.device)(z_prev, z_curr, ndrange = mod.nvar, dependencies=Event(env.device)))
                wait(copy_data_kernel(env.device)(rp_old, rp, ndrange = mod.nvar, dependencies=Event(env.device)))

                tgpu = generator_kernel_two_level(mod.gen_mod, data.baseMVA, u_curr, xbar_curr, zu_curr, lu_curr, rho_u, env.device)
                # MPI routines to be implemented:
                #  - Broadcast v_curr and l_curr to GPUs.
                #  - Collect u_curr.
                #  - div(nblk_br / number of GPUs, RoundUp)

                # time_gen += tgpu.time
                if env.use_polar
                    ev = polar_kernel_two_level(env.device)(mod.n, mod.nline, mod.line_start, mod.bus_start, scale,
                                                              u_curr, xbar_curr, zu_curr, lu_curr, rho_u,
                                                              shift_lines, env.membuf, mod.YffR, mod.YffI, mod.YftR, mod.YftI,
                                                              mod.YttR, mod.YttI, mod.YtfR, mod.YtfI, mod.FrBound, mod.ToBound, mod.brBusIdx,
                                                              ndrange=nblk_br, dependencies=Event(env.device))
                    wait(ev)
                else
                    # tgpu = CUDA.@timed @cuda threads=32 blocks=nblk_br shmem=shmem_size auglag_kernel(mod.n, inner, par.max_auglag, mod.line_start, scale, par.mu_max,
                    #                                                                                 u_curr, v_curr, l_curr, rho,
                    #                                                                                 wRIij, env.membuf, mod.YffR, mod.YffI, mod.YftR, mod.YftI,
                    #                                                                                 mod.YttR, mod.YttI, mod.YtfR, mod.YtfI, mod.FrBound, mod.ToBound)
                    error("Not implemented")
                end
                if !env.allow_infeas
                    wait(bus_kernel_two_level(env.device)(data.baseMVA, mod.nbus, mod.gen_mod.gen_start, mod.line_start, mod.bus_start,
                                                                        mod.FrStart, mod.FrIdx, mod.ToStart, mod.ToIdx, mod.GenStart, mod.GenIdx,
                                                                        mod.Pd, mod.Qd, v_curr, xbar_curr, zv_curr, lv_curr,
                                                                        rho_v, mod.YshR, mod.YshI,
                                                                        ndrange=nblk_bus, dependencies=Event(env.device)
                                                                        )
                )
                else
                    # wait(bus_kernel_two_level(env.device)(data.baseMVA, mod.nbus, mod.gen_mod.gen_start, mod.line_start, mod.bus_start,
                    #                                                     mod.FrStart, mod.FrIdx, mod.ToStart, mod.ToIdx, mod.GenStart, mod.GenIdx,
                    #                                                     mod.Pd, mod.Qd, v_curr, xbar_curr, zv_curr, lv_curr,
                    #                                                     rho_v, mod.YshR, mod.YshI,
                    #                                                     sol.s_curr, par.rho_sigma,
                    #                                                     ndrange=nblk_bus, dependencies=Event(env.device)
                    #                                                     )
                end

                # Update xbar.
                blocks=(div(mod.gen_mod.ngen-1, 64)+1)
                wait(update_xbar_generator_kernel(env.device)(mod.gen_mod.ngen, mod.gen_mod.gen_start, u_curr, v_curr,
                                               xbar_curr, zu_curr, zv_curr, lu_curr, lv_curr, rho_u, rho_v,
                                               ndrange=blocks, dependencies=Event(env.device)
                                               )
                )
                blocks=(div(mod.nline-1, 64)+1)
                wait(update_xbar_branch_kernel(env.device)(mod.nline, mod.line_start, u_curr, v_curr,
                                               xbar_curr, zu_curr, zv_curr, lu_curr, lv_curr, rho_u, rho_v,
                                               ndrange=blocks, dependencies=Event(env.device)
                                               )
                )
                blocks=(div(mod.nbus-1, 64)+1)
                wait(update_xbar_bus_kernel(env.device)(mod.nbus, mod.line_start, mod.bus_start, mod.FrStart, mod.FrIdx, mod.ToStart, mod.ToIdx, u_curr, v_curr, xbar_curr, zu_curr, zv_curr, lu_curr, lv_curr, rho_u, rho_v,
                                               ndrange=blocks, dependencies=Event(env.device)
                                               )
                )

                # Update z.
                blocks=(div(mod.gen_mod.ngen-1, 64)+1)
                wait(update_zu_generator_kernel(env.device)(mod.gen_mod.ngen, mod.gen_mod.gen_start, u_curr,
                                               xbar_curr, zu_curr, lu_curr, rho_u, lz_u, beta,
                                               ndrange=blocks, dependencies=Event(env.device)
                                               )
                )
                blocks=(div(mod.nline-1, 64)+1)
                wait(update_zu_branch_kernel(env.device)(mod.nline, mod.line_start, mod.bus_start, mod.brBusIdx, u_curr, xbar_curr, zu_curr, lu_curr, rho_u, lz_u, beta,
                                               ndrange=blocks, dependencies=Event(env.device)
                                               )
                )
                blocks=(div(mod.nvar_v-1, 64)+1)
                wait(update_zv_kernel(env.device)(mod.nvar_v, v_curr, xbar_curr, zv_curr,
                                               lv_curr, rho_v, lz_v, beta,
                                               ndrange=blocks, dependencies=Event(env.device)
                                               )
                )

                # Update multiiplier and residuals.
                blocks=(div(mod.nvar-1, 64)+1)
                wait(update_l_kernel(env.device)(mod.nvar, l_curr, z_curr, lz, beta,
                                               ndrange=blocks, dependencies=Event(env.device)
                                               )
                )
                blocks=(div(mod.gen_mod.ngen-1, 64)+1)
                wait(compute_primal_residual_u_generator_kernel(env.device)(mod.gen_mod.ngen, mod.gen_mod.gen_start, rp_u, u_curr, xbar_curr, zu_curr,
                                               ndrange=blocks, dependencies=Event(env.device)
                                               )
                )
                blocks=(div(mod.nline-1, 64)+1)
                wait(compute_primal_residual_u_branch_kernel(env.device)(mod.nline, mod.line_start, mod.bus_start, mod.brBusIdx, rp_u, u_curr, xbar_curr, zu_curr,
                                               ndrange=blocks, dependencies=Event(env.device)
                                               )
                )
                blocks=(div(mod.nvar_v-1, 64)+1)
                wait(compute_primal_residual_v_kernel(env.device)(mod.nvar_v, rp_v, v_curr, xbar_curr, zv_curr,
                                               ndrange=blocks, dependencies=Event(env.device)
                                               )
                )
                blocks=(div(mod.nvar-1, 64)+1)
                wait(vector_difference(env.device)(mod.nvar, rd, z_curr, z_prev,
                                               ndrange=blocks, dependencies=Event(env.device)
                                               )
                )

                blocks=(div(mod.nvar-1, 64)+1)
                wait(vector_difference(env.device)(mod.nvar, Ax_plus_By, rp, z_curr,
                                               ndrange=blocks, dependencies=Event(env.device)
                                               )
                )
                mismatch = norm(Ax_plus_By)

                primres = norm(rp)
                dualres = norm(rd)
                z_curr_norm = norm(z_curr)
                eps_pri = sqrt_d / (2500*outer)

                if par.verbose > 0
                    if inner == 1 || (inner % 50) == 0
                        @printf("%8s  %8s  %10s  %10s  %10s  %10s  %10s  %10s  %10s\n",
                                "Outer", "Inner", "PrimRes", "EpsPrimRes", "DualRes", "||z||", "||Ax+By||", "OuterTol", "Beta")
                    end

                    @printf("%8d  %8d  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e\n",
                            outer, inner, primres, eps_pri, dualres, z_curr_norm, mismatch, OUTER_TOL, beta)
                end

                if primres <= eps_pri || dualres <= par.DUAL_TOL
                    break
                end
            # end
        end

        if mismatch <= OUTER_TOL
            status = HAS_CONVERGED
            break
        end

        if !isa(env.device, KA.GPU)
            lz .+= max.(-par.MAX_MULTIPLIER, min.(par.MAX_MULTIPLIER, beta .* z_curr))
        else
            blocks=(div(mod.nvar-1, 64)+1)
            wait(update_lz_kernel(env.device)(mod.nvar, par.MAX_MULTIPLIER, z_curr, lz, beta,
                                               ndrange=blocks, dependencies=Event(env.device)
                                               )
            )
        end

        if z_curr_norm > theta*z_prev_norm
            beta = min(c*beta, 1e24)
        end
    end

    # Move x onto host
    hxbar_curr = Vector(xbar_curr)
    sol.objval = sum(data.generators[g].coeff[data.generators[g].n-2]*(data.baseMVA*hxbar_curr[mod.gen_mod.gen_start+2*(g-1)])^2 +
                data.generators[g].coeff[data.generators[g].n-1]*(data.baseMVA*hxbar_curr[mod.gen_mod.gen_start+2*(g-1)]) +
                data.generators[g].coeff[data.generators[g].n]
                for g in 1:mod.gen_mod.ngen)

    if outer >= outer_iterlim
        sol.status = MAXIMUM_ITERATIONS
    else
        sol.status = status
    end

    if par.verbose > 0
        # Test feasibility of global variable xbar:
        pg_err, qg_err = check_generator_bounds(env, xbar_curr)
        vm_err = check_voltage_bounds(env, xbar_curr)
        real_err, reactive_err = check_power_balance_violation(env, xbar_curr)
        @printf(" ** Violations of global variable xbar\n")
        @printf("Real power generator bounds     = %.6e\n", pg_err)
        @printf("Reactive power generator bounds = %.6e\n", qg_err)
        @printf("Voltage bounds                  = %.6e\n", vm_err)
        @printf("Real power balance              = %.6e\n", real_err)
        @printf("Reaactive power balance         = %.6e\n", reactive_err)

        rateA_nviols, rateA_maxviol, rateC_nviols, rateC_maxviol = check_linelimit_violation(data, u_curr; device = env.device)
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
        @printf("Objective value = %.6e\n", sol.objval)
    end

    return
end

function admm_rect_gpu_two_level(
    case::String;
    outer_iterlim=10, inner_iterlim=800, rho_pq=400.0, rho_va=40000.0, scale=1e-4,
    use_gpu=false, use_polar=true, allow_infeas=false, rho_sigma=1e8,
    gpu_no=0, verbose=1, outer_eps=2e-4
)
    env = AdmmEnv(
        case, use_gpu, rho_pq, rho_va; use_polar=use_polar, use_twolevel=true,
        allow_infeas=allow_infeas, rho_sigma=rho_sigma,
        gpu_no=gpu_no, verbose=verbose,
    )
    admm_restart!(env, outer_iterlim=outer_iterlim, inner_iterlim=inner_iterlim, scale=scale)
    return env
end
