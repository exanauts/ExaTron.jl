function check_generator_bounds(data, gen_start, xbar)
    ngen = length(data.generators)
    pgmin = data.genvec.Pmin
    pgmax = data.genvec.Pmax
    qgmin = data.genvec.Qmin
    qgmax = data.genvec.Qmax

    max_viol_real = 0.0
    max_viol_reactive = 0.0

    for g=1:ngen
        pidx = gen_start + 2*(g-1)
        qidx = gen_start + 2*(g-1) + 1

        real_err = max(max(0.0, xbar[pidx] - pgmax[g]), max(0.0, pgmin[g] - xbar[pidx]))
        reactive_err = max(max(0.0, xbar[qidx] - qgmax[g]), max(0.0, qgmin[g] - xbar[qidx]))

        max_viol_real = (max_viol_real < real_err) ? real_err : max_viol_real
        max_viol_reactive = (max_viol_reactive < reactive_err) ? reactive_err : max_viol_reactive
    end

    return max_viol_real, max_viol_reactive
end

function check_voltage_bounds(data, bus_start, xbar)
    nbus = length(data.buses)
    buses = data.buses

    max_viol = 0.0

    for b=1:nbus
        bidx = bus_start + 2*(b-1)
        err = max(max(0.0, xbar[bidx] - buses[b].Vmax^2), max(0.0, buses[b].Vmin^2 - xbar[bidx]))
        max_viol = (max_viol < err) ? err : max_viol
    end

    return max_viol
end

function check_power_balance_violation(data, gen_start, line_start, bus_start, xbar, YshR, YshI)
    baseMVA = data.baseMVA
    nbus = length(data.buses)

    max_viol_real = 0.0
    max_viol_reactive = 0.0
    for b=1:nbus
        real_err = 0.0
        reactive_err = 0.0
        for g in data.BusGenerators[b]
            real_err += xbar[gen_start + 2*(g-1)]
            reactive_err += xbar[gen_start + 2*(g-1)+1]
        end

        real_err -= (data.buses[b].Pd / baseMVA)
        reactive_err -= (data.buses[b].Qd / baseMVA)

        for l in data.FromLines[b]
            real_err -= xbar[line_start + 4*(l-1)]
            reactive_err -= xbar[line_start + 4*(l-1) + 1]
        end

        for l in data.ToLines[b]
            real_err -= xbar[line_start + 4*(l-1) + 2]
            reactive_err -= xbar[line_start + 4*(l-1) + 3]
        end

        real_err -= YshR[b] * xbar[bus_start + 2*(b-1)]
        reactive_err += YshI[b] * xbar[bus_start + 2*(b-1)]

        max_viol_real = (max_viol_real < abs(real_err)) ? abs(real_err) : max_viol_real
        max_viol_reactive = (max_viol_reactive < abs(reactive_err)) ? abs(reactive_err) : max_viol_reactive
    end

    return max_viol_real, max_viol_reactive
end

function get_branch_bus_index(data; use_gpu=false)
    buses = data.buses
    lines = data.lines
    BusIdx = data.BusIdx
    nline = length(lines)

    brBusIdx = [ x for l=1:nline for x in (BusIdx[lines[l].from], BusIdx[lines[l].to]) ]

    if use_gpu
        cu_brBusIdx = CuArray{Int}(undef, 2*nline)
        copyto!(cu_brBusIdx, brBusIdx)
        return cu_brBusIdx
    else
        return brBusIdx
    end
end

function init_values_two_level(data, ybus, gen_start, line_start, bus_start,
                               rho_pq, rho_va, u, v, xbar, lu, lv, rho_u, rho_v, wRIij)
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
    YshR = ybus.YshR; YshI = ybus.YshI

    rho_u .= rho_pq
    rho_v .= rho_pq
    rho_v[bus_start:end] .= rho_va

    for g=1:ngen
        pg_idx = gen_start + 2*(g-1)
        xbar[pg_idx] = 0.5*(data.genvec.Pmin[g] + data.genvec.Pmax[g])
        xbar[pg_idx+1] = 0.5*(data.genvec.Qmin[g] + data.genvec.Qmax[g])
    end

    for l=1:nline
        fr_idx = BusIdx[lines[l].from]
        to_idx = BusIdx[lines[l].to]

        wij0 = (buses[fr_idx].Vmax^2 + buses[fr_idx].Vmin^2) / 2
        wji0 = (buses[to_idx].Vmax^2 + buses[to_idx].Vmin^2) / 2
        wR0 = sqrt(wij0 * wji0)

        u_pij_idx = line_start + 8*(l-1)
        v_pij_idx = line_start + 4*(l-1)
        v[v_pij_idx] = u[u_pij_idx] = YffR[l] * wij0 + YftR[l] * wR0
        v[v_pij_idx+1] = u[u_pij_idx+1] = -YffI[l] * wij0 - YftI[l] * wR0
        v[v_pij_idx+2] = u[u_pij_idx+2] = YttR[l] * wji0 + YtfR[l] * wR0
        v[v_pij_idx+3] = u[u_pij_idx+3] = -YttI[l] * wji0 - YtfI[l] * wR0

        rho_u[u_pij_idx+4:u_pij_idx+7] .= rho_va

        wRIij[2*(l-1)+1] = wR0
        wRIij[2*l] = 0.0
    end

    for b=1:nbus
        xbar[bus_start + 2*(b-1)] = (buses[b].Vmax^2 + buses[b].Vmin^2) / 2
        xbar[bus_start + 2*(b-1)+1] = 0.0
    end

    lu .= 0.0
    lv .= 0.0

    return
end

function update_xbar(data, gen_start, line_start, bus_start, FrStart, ToStart, FrIdx, ToIdx,
                     u, v, xbar, zu, zv, lu, lv, rho_u, rho_v)
    lines = data.lines
    BusIdx = data.BusIdx
    ngen = length(data.generators)
    nline = length(data.lines)
    nbus = length(data.buses)

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

function update_xbar_generator_kernel(n, gen_start, u, v, xbar, zu, zv, lu, lv, rho_u, rho_v)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if tx <= n
        i = gen_start + 2*(tx - 1)
        @inbounds begin
            xbar[i] = (lu[i] + rho_u[i]*(u[i] + zu[i]) + lv[i] + rho_v[i]*(v[i] + zv[i])) / (rho_u[i] + rho_v[i])
            xbar[i+1] = (lu[i+1] + rho_u[i+1]*(u[i+1] + zu[i+1]) + lv[i+1] + rho_v[i+1]*(v[i+1] + zv[i+1])) / (rho_u[i+1] + rho_v[i+1])
        end
    end

    return
end

function update_xbar_branch_kernel(n, line_start, u, v, xbar, zu, zv, lu, lv, rho_u, rho_v)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

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

    return
end

function update_xbar_bus_kernel(n, line_start, bus_start, FrStart, FrIdx, ToStart, ToIdx, u, v, xbar, zu, zv, lu, lv, rho_u, rho_v)
    b = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

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

    return
end

function update_lu(data, gen_start, line_start, bus_start, u, xbar, zu, l, rho)
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

function update_zu(data, gen_start, line_start, bus_start, u, xbar, z, l, rho, lz, beta)
    lines = data.lines
    BusIdx = data.BusIdx
    ngen = length(data.generators)
    nline = length(data.lines)

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

function update_zu_generator_kernel(n, gen_start, u, xbar, z, l, rho, lz, beta)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if tx <= n
        i = gen_start  + 2*(tx - 1)
        @inbounds begin
            z[i] = (-(lz[i] + l[i] + rho[i]*(u[i] - xbar[i]))) / (beta + rho[i])
            z[i+1] = (-(lz[i+1] + l[i+1] + rho[i+1]*(u[i+1] - xbar[i+1]))) / (beta + rho[i+1])
        end
    end

    return
end

function update_zu_branch_kernel(n, line_start, bus_start, brBusIdx, u, xbar, z, l, rho, lz, beta)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

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

    return
end

function update_zv_kernel(n, v, xbar, z, l, rho, lz, beta)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if tx <= n
        @inbounds begin
            z[tx] = (-(lz[tx] + l[tx] + rho[tx]*(v[tx] - xbar[tx]))) / (beta + rho[tx])
        end
    end

    return
end

function update_l_kernel(n, l, z, lz, beta)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if tx <= n
        @inbounds begin
            l[tx] = -(lz[tx] + beta*z[tx])
        end
    end

    return
end

function compute_primal_residual_u(data, gen_start, line_start, bus_start, rp_u, u, xbar, z)
    lines = data.lines
    BusIdx = data.BusIdx
    ngen = length(data.generators)
    nline = length(data.lines)

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

function compute_primal_residual_u_generator_kernel(n, gen_start, rp, u, xbar, z)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if tx <= n
        i = gen_start + 2*(tx - 1)
        @inbounds begin
            rp[i] = u[i] - xbar[i] + z[i]
            rp[i+1] = u[i+1] - xbar[i+1] + z[i+1]
        end
    end

    return
end

function compute_primal_residual_u_branch_kernel(n, line_start, bus_start, brBusIdx, rp, u, xbar, z)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

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

    return
end

function compute_primal_residual_v_kernel(n, rp, v, xbar, z)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if tx <= n
        rp[tx] = v[tx] - xbar[tx] + z[tx]
    end

    return
end

function vector_difference(n, c, a, b)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if tx <= n
        c[tx] = a[tx] - b[tx]
    end

    return
end

function update_lz_kernel(n, max_limit, z, lz, beta)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if tx <= n
        lz[tx] += max(-max_limit, min(max_limit, beta*z[tx]))
    end

    return
end

function update_rho(rho, rp, rp_old, theta, gamma)
    for i=1:length(rho)
        if abs(rp[i]) > theta*abs(rp_old[i])
            rho[i] = min(gamma*rho[i], 1e24)
        end
    end
end

function admm_rect_gpu_two_level(case; outer_iterlim=10, inner_iterlim=800, rho_pq=400.0, rho_va=40000.0, scale=1e-4,
                                 use_gpu=false, use_polar=true, gpu_no=1, outer_eps=2*1e-4)
    data = opf_loaddata(case)

    ngen = length(data.generators)
    nline = length(data.lines)
    nbus = length(data.buses)

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

    nvar_u = 2*ngen + 8*nline
    nvar_v = 2*ngen + 4*nline + 2*nbus
    nvar = nvar_u + nvar_v
    gen_start = 1
    line_start = 2*ngen + 1
    bus_start = 2*ngen + 4*nline + 1 # this is for varibles of type v.

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
    brBusIdx = get_branch_bus_index(data)

    cu_pgmin, cu_pgmax, cu_qgmin, cu_qgmax, cu_c2, cu_c1, cu_c0 = get_generator_data(data; use_gpu=use_gpu)
    cuYshR, cuYshI, cuYffR, cuYffI, cuYftR, cuYftI, cuYttR, cuYttI, cuYtfR, cuYtfI, cuFrBound, cuToBound = get_branch_data(data; use_gpu=use_gpu)
    cu_FrStart, cu_FrIdx, cu_ToStart, cu_ToIdx, cu_GenStart, cu_GenIdx, cu_Pd, cu_Qd = get_bus_data(data; use_gpu=use_gpu)
    cu_brBusIdx = get_branch_bus_index(data; use_gpu=use_gpu)

    x_curr = zeros(nvar)
    xbar_curr = zeros(nvar_v)
    z_outer = zeros(nvar)
    z_curr = zeros(nvar)
    z_prev = zeros(nvar)
    l_curr = zeros(nvar)
    lz = zeros(nvar)
    rho = zeros(nvar)
    rd = zeros(nvar)
    rp = zeros(nvar)
    rp_old = zeros(nvar)
    Ax_plus_By = zeros(nvar)
    param = zeros(31, nline)
    wRIij = zeros(2*nline)

    u_curr = view(x_curr, 1:nvar_u)
    v_curr = view(x_curr, nvar_u+1:nvar)
    zu_outer = view(z_outer, 1:nvar_u)
    zv_outer = view(z_outer, nvar_u+1:nvar)
    zu_curr = view(z_curr, 1:nvar_u)
    zv_curr = view(z_curr, nvar_u+1:nvar)
    zu_prev = view(z_prev, 1:nvar_u)
    zv_prev = view(z_curr, nvar_u+1:nvar)
    lu_curr = view(l_curr, 1:nvar_u)
    lv_curr = view(l_curr, nvar_u+1:nvar)
    lz_u = view(lz, 1:nvar_u)
    lz_v = view(lz, nvar_u+1:nvar)
    rho_u = view(rho, 1:nvar_u)
    rho_v = view(rho, nvar_u+1:nvar)
    rp_u = view(rp, 1:nvar_u)
    rp_v = view(rp, nvar_u+1:nvar)
    rd_u = view(rd, 1:nvar_u)
    rd_v = view(rd, nvar_u+1:nvar)

    init_values_two_level(data, ybus, gen_start, line_start, bus_start,
                          rho_pq, rho_va, u_curr, v_curr, xbar_curr, lu_curr, lv_curr, rho_u, rho_v, wRIij)

    if use_gpu
        cu_x_curr = CuArray{Float64}(undef, nvar)
        cu_xbar_curr = CuArray{Float64}(undef, nvar_v)
        cu_z_outer = CuArray{Float64}(undef, nvar)
        cu_z_curr = CuArray{Float64}(undef, nvar)
        cu_z_prev = CuArray{Float64}(undef, nvar)
        cu_l_curr = CuArray{Float64}(undef, nvar)
        cu_lz = CuArray{Float64}(undef, nvar)
        cu_rho = CuArray{Float64}(undef, nvar)
        cu_rd = CuArray{Float64}(undef, nvar)
        cu_rp = CuArray{Float64}(undef, nvar)
        cu_rp_old = CuArray{Float64}(undef, nvar)
        cu_Ax_plus_By = CuArray{Float64}(undef, nvar)
        cuParam = CuArray{Float64}(undef, (31, nline))
        cuWRIij = CuArray{Float64}(undef, 2*nline)

        cu_u_curr = view(cu_x_curr, 1:nvar_u)
        cu_v_curr = view(cu_x_curr, nvar_u+1:nvar)
        cu_zu_curr = view(cu_z_curr, 1:nvar_u)
        cu_zv_curr = view(cu_z_curr, nvar_u+1:nvar)
        cu_lu_curr = view(cu_l_curr, 1:nvar_u)
        cu_lv_curr = view(cu_l_curr, nvar_u+1:nvar)
        cu_lz_u = view(cu_lz, 1:nvar_u)
        cu_lz_v = view(cu_lz, nvar_u+1:nvar)
        cu_rho_u = view(cu_rho, 1:nvar_u)
        cu_rho_v = view(cu_rho, nvar_u+1:nvar)
        cu_rp_u = view(cu_rp, 1:nvar_u)
        cu_rp_v = view(cu_rp, nvar_u+1:nvar)

        copyto!(cu_x_curr, x_curr)
        copyto!(cu_xbar_curr, xbar_curr)
        copyto!(cu_z_outer, z_outer)
        copyto!(cu_z_curr, z_curr)
        copyto!(cu_l_curr, l_curr)
        copyto!(cu_lz, lz)
        copyto!(cu_rho, rho)
        copyto!(cu_Ax_plus_By, Ax_plus_By)
        copyto!(cuParam, param)
        copyto!(cuWRIij, wRIij)
    end

    max_auglag = 50

    nblk_gen = div(ngen, 32, RoundUp)
    nblk_br = nline
    nblk_bus = div(nbus, 32, RoundUp)

    ABSTOL = 1e-6
    RELTOL = 1e-5

    beta = 1e3
    gamma = 6.0
    c = 6.0
    theta = 0.8
    sqrt_d = sqrt(nvar_u + nvar_v)

    MAX_MULTIPLIER = 1e12
    DUAL_TOL = 1e-8
    OUTER_TOL = sqrt_d*(outer_eps)

    outer = 0
    inner = 0

    time_gen = time_br = time_bus = 0
    shift_lines = 0
    shmem_size = sizeof(Float64)*(14*n+3*n^2) + sizeof(Int)*(4*n)

    mismatch = Inf
    z_prev_norm = z_curr_norm = Inf

    @time begin
    while outer < outer_iterlim
        outer += 1

        if !use_gpu
            z_outer .= z_curr
            z_prev_norm = norm(z_outer)
        else
            @cuda threads=64 blocks=(div(nvar-1, 64)+1) copy_data_kernel(nvar, cu_z_outer, cu_z_curr)
            z_prev_norm = CUDA.norm(cu_z_curr)
            CUDA.synchronize()
        end

        inner = 0
        while inner < inner_iterlim
            inner += 1

            if !use_gpu
                z_prev .= z_curr
                rp_old .= rp

                tcpu = @timed generator_kernel_two_level_cpu(baseMVA, ngen, gen_start, u_curr, xbar_curr, zu_curr, lu_curr, rho_u,
                                                            pgmin, pgmax, qgmin, qgmax, c2, c1)
                time_gen += tcpu.time

                #scale = min(scale, (2*1e4) / maximum(abs.(rho_u)))
                if use_polar
                    tcpu = @timed auglag_it, tron_it = polar_kernel_two_level_cpu(n, nline, line_start, bus_start, scale,
                                                                                u_curr, xbar_curr, zu_curr, lu_curr, rho_u,
                                                                                shift_lines, param, YffR, YffI, YftR, YftI,
                                                                                YttR, YttI, YtfR, YtfI, FrBound, ToBound, brBusIdx)
                else
                    tcpu = @timed auglag_it, tron_it = auglag_kernel_cpu(n, nline, inner, max_auglag, line_start, mu_max,
                                                                        u_curr, v_curr, l_curr, rho,
                                                                        wRIij, param, YffR, YffI, YftR, YftI,
                                                                        YttR, YttI, YtfR, YtfI, FrBound, ToBound)
                end
                time_br += tcpu.time

                tcpu = @timed bus_kernel_two_level_cpu(baseMVA, nbus, gen_start, line_start, bus_start,
                                                    FrStart, FrIdx, ToStart, ToIdx, GenStart, GenIdx,
                                                    Pd, Qd, v_curr, xbar_curr, zv_curr, lv_curr, rho_v, YshR, YshI)
                time_bus += tcpu.time

                update_xbar(data, gen_start, line_start, bus_start, FrStart, ToStart, FrIdx, ToIdx,
                            u_curr, v_curr, xbar_curr, zu_curr, zv_curr, lu_curr, lv_curr, rho_u, rho_v)

                update_zu(data, gen_start, line_start, bus_start, u_curr, xbar_curr, zu_curr, lu_curr, rho_u, lz_u, beta)
                zv_curr .= (-(lz_v .+ lv_curr .+ rho_v.*(v_curr .- xbar_curr))) ./ (beta .+ rho_v)

                l_curr .= -(lz .+ beta.*z_curr)

                compute_primal_residual_u(data, gen_start, line_start, bus_start, rp_u, u_curr, xbar_curr, zu_curr)
                rp_v .= v_curr .- xbar_curr .+ zv_curr

                #=
                if inner > 1
                    update_rho(rho, rp, rp_old, theta, gamma)
                end
                =#

                rd .= z_curr .- z_prev
                Ax_plus_By .= rp .- z_curr

                primres = norm(rp)
                dualres = norm(rd)
                z_curr_norm = norm(z_curr)
                mismatch = norm(Ax_plus_By)
                eps_pri = sqrt_d / (2500*outer)

                if inner == 1 || (inner % 50) == 0
                    @printf("%8s  %8s  %10s  %10s  %10s  %10s  %10s  %10s\n",
                            "Outer", "Inner", "PrimRes", "EpsPrimRes", "DualRes", "||z||", "||Ax+By||", "Beta")
                end

                @printf("%8d  %8d  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e\n",
                        outer, inner, primres, eps_pri, dualres, z_curr_norm, mismatch, beta)

                if primres <= eps_pri || dualres <= DUAL_TOL
                    break
                end
            else
                @cuda threads=64 blocks=(div(nvar-1, 64)+1) copy_data_kernel(nvar, cu_z_prev, cu_z_curr)
                @cuda threads=64 blocks=(div(nvar-1, 64)+1) copy_data_kernel(nvar, cu_rp_old, cu_rp)
                CUDA.synchronize()

                tgpu = CUDA.@timed @cuda threads=32 blocks=nblk_gen generator_kernel_two_level(baseMVA, ngen, gen_start,
                                                              cu_u_curr, cu_xbar_curr, cu_zu_curr, cu_lu_curr, cu_rho_u,
                                                              cu_pgmin, cu_pgmax, cu_qgmin, cu_qgmax, cu_c2, cu_c1)
                # MPI routines to be implemented:
                #  - Broadcast cu_v_curr and cu_l_curr to GPUs.
                #  - Collect cu_u_curr.
                #  - div(nblk_br / number of GPUs, RoundUp)

                time_gen += tgpu.time
                if use_polar
                    tgpu = CUDA.@timed @cuda threads=32 blocks=nblk_br shmem=shmem_size polar_kernel_two_level(n, nline, line_start, bus_start, scale,
                                                              cu_u_curr, cu_xbar_curr, cu_zu_curr, cu_lu_curr, cu_rho_u,
                                                              shift_lines, cuParam, cuYffR, cuYffI, cuYftR, cuYftI,
                                                              cuYttR, cuYttI, cuYtfR, cuYtfI, cuFrBound, cuToBound, cu_brBusIdx)
                else
                    tgpu = CUDA.@timed @cuda threads=32 blocks=nblk_br shmem=shmem_size auglag_kernel(n, inner, max_auglag, line_start, scale, mu_max,
                                                                                                    cu_u_curr, cu_v_curr, cu_l_curr, cu_rho,
                                                                                                    cuWRIij, cuParam, cuYffR, cuYffI, cuYftR, cuYftI,
                                                                                                    cuYttR, cuYttI, cuYtfR, cuYtfI, cuFrBound, cuToBound)
                end
                time_br += tgpu.time
                tgpu = CUDA.@timed @cuda threads=32 blocks=nblk_bus bus_kernel_two_level(baseMVA, nbus, gen_start, line_start, bus_start,
                                                                     cu_FrStart, cu_FrIdx, cu_ToStart, cu_ToIdx, cu_GenStart, cu_GenIdx,
                                                                     cu_Pd, cu_Qd, cu_v_curr, cu_xbar_curr, cu_zv_curr, cu_lv_curr,
                                                                     cu_rho_v, cuYshR, cuYshI)
                time_bus += tgpu.time

                # Update xbar.
                @cuda threads=64 blocks=(div(ngen-1, 64)+1) update_xbar_generator_kernel(ngen, gen_start, cu_u_curr, cu_v_curr,
                                               cu_xbar_curr, cu_zu_curr, cu_zv_curr, cu_lu_curr, cu_lv_curr, cu_rho_u, cu_rho_v)
                @cuda threads=64 blocks=(div(nline-1, 64)+1) update_xbar_branch_kernel(nline, line_start, cu_u_curr, cu_v_curr,
                                               cu_xbar_curr, cu_zu_curr, cu_zv_curr, cu_lu_curr, cu_lv_curr, cu_rho_u, cu_rho_v)
                @cuda threads=64 blocks=(div(nbus-1, 64)+1) update_xbar_bus_kernel(nbus, line_start, bus_start, cu_FrStart, cu_FrIdx, cu_ToStart, cu_ToIdx,
                                               cu_u_curr, cu_v_curr, cu_xbar_curr, cu_zu_curr, cu_zv_curr, cu_lu_curr, cu_lv_curr, cu_rho_u, cu_rho_v)
                CUDA.synchronize()

                # Update z.
                @cuda threads=64 blocks=(div(ngen-1, 64)+1) update_zu_generator_kernel(ngen, gen_start, cu_u_curr,
                                               cu_xbar_curr, cu_zu_curr, cu_lu_curr, cu_rho_u, cu_lz_u, beta)
                @cuda threads=64 blocks=(div(nline-1, 64)+1) update_zu_branch_kernel(nline, line_start, bus_start, cu_brBusIdx,
                                               cu_u_curr, cu_xbar_curr, cu_zu_curr, cu_lu_curr, cu_rho_u, cu_lz_u, beta)
                @cuda threads=64 blocks=(div(nvar_v-1, 64)+1) update_zv_kernel(nvar_v, cu_v_curr, cu_xbar_curr, cu_zv_curr,
                                               cu_lv_curr, cu_rho_v, cu_lz_v, beta)
                CUDA.synchronize()

                # Update multiiplier and residuals.
                @cuda threads=64 blocks=(div(nvar-1, 64)+1) update_l_kernel(nvar, cu_l_curr, cu_z_curr, cu_lz, beta)
                @cuda threads=64 blocks=(div(ngen-1, 64)+1) compute_primal_residual_u_generator_kernel(ngen, gen_start, cu_rp_u, cu_u_curr, cu_xbar_curr, cu_zu_curr)
                @cuda threads=64 blocks=(div(nline-1, 64)+1) compute_primal_residual_u_branch_kernel(nline, line_start, bus_start, cu_brBusIdx, cu_rp_u, cu_u_curr, cu_xbar_curr, cu_zu_curr)
                @cuda threads=64 blocks=(div(nvar_v-1, 64)+1) compute_primal_residual_v_kernel(nvar_v, cu_rp_v, cu_v_curr, cu_xbar_curr, cu_zv_curr)
                @cuda threads=64 blocks=(div(nvar-1, 64)+1) vector_difference(nvar, cu_rd, cu_z_curr, cu_z_prev)
                CUDA.synchronize()

                CUDA.@sync @cuda threads=64 blocks=(div(nvar-1, 64)+1) vector_difference(nvar, cu_Ax_plus_By, cu_rp, cu_z_curr)
                mismatch = CUDA.norm(cu_Ax_plus_By)

                primres = CUDA.norm(cu_rp)
                dualres = CUDA.norm(cu_rd)
                z_curr_norm = CUDA.norm(cu_z_curr)
                eps_pri = sqrt_d / (2500*outer)

                if inner == 1 || (inner % 50) == 0
                    @printf("%8s  %8s  %10s  %10s  %10s  %10s  %10s  %10s  %10s\n",
                            "Outer", "Inner", "PrimRes", "EpsPrimRes", "DualRes", "||z||", "||Ax+By||", "OuterTol", "Beta")
                end

                @printf("%8d  %8d  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e\n",
                        outer, inner, primres, eps_pri, dualres, z_curr_norm, mismatch, OUTER_TOL, beta)

                if primres <= eps_pri || dualres <= DUAL_TOL
                    break
                end
            end
        end

        if mismatch <= OUTER_TOL
            break
        end

        if !use_gpu
            lz .+= max.(-MAX_MULTIPLIER, min.(MAX_MULTIPLIER, beta .* z_curr))
        else
            CUDA.@sync @cuda threads=64 blocks=(div(nvar-1, 64)+1) update_lz_kernel(nvar, MAX_MULTIPLIER, cu_z_curr, cu_lz, beta)
        end

        if z_curr_norm > theta*z_prev_norm
            beta = min(c*beta, 1e24)
        end
    end
    end

    if use_gpu
        # Copying from view to view generates "Warning: Performing scalar operations on GPU arrays."
        # We ignore this seemingly wrong message for now.
        copyto!(x_curr, 1, cu_x_curr, 1, nvar_u)
        copyto!(x_curr, nvar_u+1, cu_x_curr, nvar_u+1, nvar_v)
        copyto!(xbar_curr, cu_xbar_curr)
    end

    # Test feasibility of global variable xbar:
    pg_err, qg_err = check_generator_bounds(data, gen_start, xbar_curr)
    vm_err = check_voltage_bounds(data, bus_start, xbar_curr)
    real_err, reactive_err = check_power_balance_violation(data, gen_start, line_start, bus_start, xbar_curr, YshR, YshI)
    @printf(" ** Violations of global variable xbar\n")
    @printf("Real power generator bounds     = %.6e\n", pg_err)
    @printf("Reactive power generator bounds = %.6e\n", qg_err)
    @printf("Voltage bounds                  = %.6e\n", vm_err)
    @printf("Real power balance              = %.6e\n", real_err)
    @printf("Reaactive power balance         = %.6e\n", reactive_err)

    rateA_nviols, rateA_maxviol, rateC_nviols, rateC_maxviol = check_linelimit_violation(data, u_curr)
    @printf(" ** Line limit violations\n")
    @printf("RateA number of violations = %d (%d)\n", rateA_nviols, nline)
    @printf("RateA maximum violation    = %.2f\n", rateA_maxviol)
    @printf("RateC number of violations = %d (%d)\n", rateC_nviols, nline)
    @printf("RateC maximum violation    = %.2f\n", rateC_maxviol)

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