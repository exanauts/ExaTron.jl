function check_generator_bounds(model::Model, xbar)
    ngen = model.ngen
    gen_start = model.gen_start

    pgmax = model.pgmax; pgmin = model.pgmin
    qgmax = model.qgmax; qgmin = model.qgmin

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

function check_voltage_bounds(model::Model, xbar)
    nbus = model.nbus
    bus_start = model.bus_start

    max_viol = 0.0

    for b=1:nbus
        bidx = bus_start + 2*(b-1)
        err = max(max(0.0, xbar[bidx] - model.Vmax[b]^2), max(0.0, model.Vmin[b]^2 - xbar[bidx]))
        max_viol = (max_viol < err) ? err : max_viol
    end

    return max_viol
end

function check_power_balance_violation(model::Model, xbar)
    baseMVA = model.baseMVA
    nbus = model.nbus
    gen_start, line_start, bus_start, YshR, YshI = model.gen_start, model.line_start, model.bus_start, model.YshR, model.YshI

    max_viol_real = 0.0
    max_viol_reactive = 0.0
    for b=1:nbus
        real_err = 0.0
        reactive_err = 0.0
        for k=model.GenStart[b]:model.GenStart[b+1]-1
            g = model.GenIdx[k]
            real_err += xbar[gen_start + 2*(g-1)]
            reactive_err += xbar[gen_start + 2*(g-1)+1]
        end

        real_err -= (model.Pd[b] / baseMVA)
        reactive_err -= (model.Qd[b] / baseMVA)
        #real_err -= (data.buses[b].Pd / baseMVA)
        #reactive_err -= (data.buses[b].Qd / baseMVA)

        for k=model.FrStart[b]:model.FrStart[b+1]-1
            l = model.FrIdx[k]
            real_err -= xbar[line_start + 4*(l-1)]
            reactive_err -= xbar[line_start + 4*(l-1) + 1]
        end

        for k=model.ToStart[b]:model.ToStart[b+1]-1
            l = model.ToIdx[k]
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

function get_branch_bus_index(data::OPFData; use_gpu=false)
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

function init_solution!(model::Model{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
                        sol::SolutionTwoLevel{Float64,Array{Float64,1}}, rho_pq, rho_va)
    gen_start, line_start, bus_start = model.gen_start, model.line_start, model.bus_start

    u_curr = view(sol.x_curr, 1:model.nvar_u)
    v_curr = view(sol.x_curr, model.nvar_u+1:model.nvar)
    rho_u = view(sol.rho, 1:model.nvar_u)
    rho_v = view(sol.rho, model.nvar_u+1:model.nvar)

    rho_u .= rho_pq
    rho_v .= rho_pq
    rho_v[bus_start:end] .= rho_va

    for g=1:model.ngen
        pg_idx = gen_start + 2*(g-1)
        sol.xbar_curr[pg_idx] = 0.5*(model.pgmin[g] + model.pgmax[g])
        sol.xbar_curr[pg_idx+1] = 0.5*(model.qgmin[g] + model.qgmax[g])
    end

    fill!(sol.wRIij, 0.0)
    for l=1:model.nline
        fr_idx = model.brBusIdx[2*(l-1)+1]
        to_idx = model.brBusIdx[2*l]

        wij0 = (model.Vmax[fr_idx]^2 + model.Vmin[fr_idx]^2) / 2
        wji0 = (model.Vmax[to_idx]^2 + model.Vmin[to_idx]^2) / 2
        wR0 = sqrt(wij0 * wji0)

        u_pij_idx = line_start + 8*(l-1)
        v_pij_idx = line_start + 4*(l-1)
        v_curr[v_pij_idx] = u_curr[u_pij_idx] = model.YffR[l] * wij0 + model.YftR[l] * wR0
        v_curr[v_pij_idx+1] = u_curr[u_pij_idx+1] = -model.YffI[l] * wij0 - model.YftI[l] * wR0
        v_curr[v_pij_idx+2] = u_curr[u_pij_idx+2] = model.YttR[l] * wji0 + model.YtfR[l] * wR0
        v_curr[v_pij_idx+3] = u_curr[u_pij_idx+3] = -model.YttI[l] * wji0 - model.YtfI[l] * wR0

        rho_u[u_pij_idx+4:u_pij_idx+7] .= rho_va

        sol.wRIij[2*(l-1)+1] = wR0
        sol.wRIij[2*l] = 0.0
    end

    for b=1:model.nbus
        sol.xbar_curr[bus_start + 2*(b-1)] = (model.Vmax[b]^2 + model.Vmin[b]^2) / 2
        sol.xbar_curr[bus_start + 2*(b-1)+1] = 0.0
    end

    sol.l_curr .= 0.0

    return
end

function init_generator_kernel_two_level(n::Int, gen_start::Int,
    pgmax::CuDeviceArray{Float64,1}, pgmin::CuDeviceArray{Float64,1},
    qgmax::CuDeviceArray{Float64,1}, qgmin::CuDeviceArray{Float64,1},
    xbar::CuDeviceArray{Float64,1})

    g = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if g <= n
        xbar[gen_start + 2*(g-1)] = 0.5*(pgmin[g] + pgmax[g])
        xbar[gen_start + 2*(g-1)+1] = 0.5*(qgmin[g] + qgmax[g])
    end

    return
end

function init_branch_kernel_two_level(n::Int, line_start::Int, rho_va::Float64,
    u_curr::CuDeviceArray{Float64,1}, v_curr::CuDeviceArray{Float64,1},
    rho_u::CuDeviceArray{Float64,1}, wRIij::CuDeviceArray{Float64,1},
    brBusIdx::CuDeviceArray{Int,1},
    Vmax::CuDeviceArray{Float64,1}, Vmin::CuDeviceArray{Float64,1},
    YffR::CuDeviceArray{Float64,1}, YffI::CuDeviceArray{Float64,1},
    YftR::CuDeviceArray{Float64,1}, YftI::CuDeviceArray{Float64,1},
    YtfR::CuDeviceArray{Float64,1}, YtfI::CuDeviceArray{Float64,1},
    YttR::CuDeviceArray{Float64,1}, YttI::CuDeviceArray{Float64,1})

    l = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if l <= n
        fr_idx = brBusIdx[2*(l-1)+1]
        to_idx = brBusIdx[2*l]

        wij0 = (Vmax[fr_idx]^2 + Vmin[fr_idx]^2) / 2
        wji0 = (Vmax[to_idx]^2 + Vmin[to_idx]^2) / 2
        wR0 = sqrt(wij0 * wji0)

        u_pij_idx = line_start + 8*(l-1)
        v_pij_idx = line_start + 4*(l-1)
        v_curr[v_pij_idx] = u_curr[u_pij_idx] = YffR[l] * wij0 + YftR[l] * wR0
        v_curr[v_pij_idx+1] = u_curr[u_pij_idx+1] = -YffI[l] * wij0 - YftI[l] * wR0
        v_curr[v_pij_idx+2] = u_curr[u_pij_idx+2] = YttR[l] * wji0 + YtfR[l] * wR0
        v_curr[v_pij_idx+3] = u_curr[u_pij_idx+3] = -YttI[l] * wji0 - YtfI[l] * wR0

        rho_u[u_pij_idx+4:u_pij_idx+7] .= rho_va

        wRIij[2*(l-1)+1] = wR0
        wRIij[2*l] = 0.0
    end

    return
end

function init_bus_kernel_two_level(n::Int, bus_start::Int,
    Vmax::CuDeviceArray{Float64,1}, Vmin::CuDeviceArray{Float64,1}, xbar::CuDeviceArray{Float64,1})
    b = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if b <= n
        xbar[bus_start + 2*(b-1)] = (Vmax[b]^2 + Vmin[b]^2) / 2
        xbar[bus_start + 2*(b-1)+1] = 0.0
    end

    return
end

function init_solution!(model::Model{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
                        sol::SolutionTwoLevel{Float64,CuArray{Float64,1}}, rho_pq, rho_va)
    gen_start, line_start, bus_start = model.gen_start, model.line_start, model.bus_start

    u_curr = view(sol.x_curr, 1:model.nvar_u)
    v_curr = view(sol.x_curr, model.nvar_u+1:model.nvar)
    rho_u = view(sol.rho, 1:model.nvar_u)
    rho_v = view(sol.rho, model.nvar_u+1:model.nvar)

    rho_u .= rho_pq
    rho_v .= rho_pq
    rho_v[bus_start:end] .= rho_va

    fill!(sol.wRIij, 0.0)
    sol.l_curr .= 0.0

    @cuda threads=64 blocks=(div(model.ngen-1,64)+1) init_generator_kernel_two_level(model.ngen, gen_start,
                    model.pgmax, model.pgmin, model.qgmax, model.qgmin, sol.xbar_curr)
    @cuda threads=64 blocks=(div(model.nline-1,64)+1) init_branch_kernel_two_level(model.nline, line_start,
                    rho_va, u_curr, v_curr, rho_u, sol.wRIij,
                    model.brBusIdx, model.Vmax, model.Vmin, model.YffR, model.YffI,
                    model.YftR, model.YftI, model.YtfR, model.YtfI, model.YttR, model.YttI)
    @cuda threads=64 blocks=(div(model.nbus-1,64)+1) init_bus_kernel_two_level(model.nbus, bus_start,
                    model.Vmax, model.Vmin, sol.xbar_curr)
    CUDA.synchronize()

    return
end

#=
function init_solution!(env::AdmmEnv, sol::SolutionTwoLevel, ybus::Ybus, rho_pq, rho_va)
    if env.is_multiperiod
        data, model = env.data, env.model.single_period
    else
        data, model = env.data, env.model
    end

    gen_start, line_start, bus_start = model.gen_start, model.line_start, model.bus_start
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

    u_curr = view(sol.x_curr, 1:model.nvar_u)
    v_curr = view(sol.x_curr, model.nvar_u+1:model.nvar)
    rho_u = view(sol.rho, 1:model.nvar_u)
    rho_v = view(sol.rho, model.nvar_u+1:model.nvar)

    rho_u .= rho_pq
    rho_v .= rho_pq
    rho_v[bus_start:end] .= rho_va

    for g=1:ngen
        pg_idx = gen_start + 2*(g-1)
        sol.xbar_curr[pg_idx] = 0.5*(data.generators[g].Pmin + data.generators[g].Pmax)
        sol.xbar_curr[pg_idx+1] = 0.5*(data.generators[g].Qmin + data.generators[g].Qmax)
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
        v_curr[v_pij_idx] = u_curr[u_pij_idx] = YffR[l] * wij0 + YftR[l] * wR0
        v_curr[v_pij_idx+1] = u_curr[u_pij_idx+1] = -YffI[l] * wij0 - YftI[l] * wR0
        v_curr[v_pij_idx+2] = u_curr[u_pij_idx+2] = YttR[l] * wji0 + YtfR[l] * wR0
        v_curr[v_pij_idx+3] = u_curr[u_pij_idx+3] = -YttI[l] * wji0 - YtfI[l] * wR0

        rho_u[u_pij_idx+4:u_pij_idx+7] .= rho_va

        sol.wRIij[2*(l-1)+1] = wR0
        sol.wRIij[2*l] = 0.0
    end

    for b=1:nbus
        sol.xbar_curr[bus_start + 2*(b-1)] = (buses[b].Vmax^2 + buses[b].Vmin^2) / 2
        sol.xbar_curr[bus_start + 2*(b-1)+1] = 0.0
    end

    sol.l_curr .= 0.0

    return
end
=#

function update_xbar(model::Model, u, v, xbar, zu, zv, lu, lv, rho_u, rho_v)
    ngen = model.ngen
    nline = model.nline
    nbus = model.nbus

    gen_start = model.gen_start
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

function update_xbar_generator_kernel(n::Int, gen_start::Int, u, v, xbar, zu, zv, lu, lv, rho_u, rho_v)
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

function update_xbar_branch_kernel(n::Int, line_start::Int, u, v, xbar, zu, zv, lu, lv, rho_u, rho_v)
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

function update_xbar_bus_kernel(n::Int, line_start::Int, bus_start::Int, FrStart, FrIdx, ToStart, ToIdx, u, v, xbar, zu, zv, lu, lv, rho_u, rho_v)
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

function update_zu(model::Model, u, xbar, z, l, rho, lz, beta)
    ngen = model.ngen
    nline = model.nline
    brBusIdx = model.brBusIdx
    gen_start, line_start, bus_start = model.gen_start, model.line_start, model.bus_start

    gen_end = gen_start + 2*ngen - 1
    z[gen_start:gen_end] .= (-(lz[gen_start:gen_end] .+ l[gen_start:gen_end] .+ rho[gen_start:gen_end].*(u[gen_start:gen_end] .- xbar[gen_start:gen_end]))) ./ (beta .+ rho[gen_start:gen_end])

    ul_cur = line_start
    xl_cur = line_start
    for j=1:nline
        fr_idx = bus_start + 2*(brBusIdx[2*(j-1)+1]-1)
        to_idx = bus_start + 2*(brBusIdx[2*j]-1)

        z[ul_cur:ul_cur+3] .= (-(lz[ul_cur:ul_cur+3] .+ l[ul_cur:ul_cur+3] .+ rho[ul_cur:ul_cur+3].*(u[ul_cur:ul_cur+3] .- xbar[xl_cur:xl_cur+3]))) ./ (beta .+ rho[ul_cur:ul_cur+3])
        z[ul_cur+4] = (-(lz[ul_cur+4] + l[ul_cur+4] + rho[ul_cur+4]*(u[ul_cur+4] - xbar[fr_idx]))) / (beta + rho[ul_cur+4])
        z[ul_cur+5] = (-(lz[ul_cur+5] + l[ul_cur+5] + rho[ul_cur+5]*(u[ul_cur+5] - xbar[to_idx]))) / (beta + rho[ul_cur+5])
        z[ul_cur+6] = (-(lz[ul_cur+6] + l[ul_cur+6] + rho[ul_cur+6]*(u[ul_cur+6] - xbar[fr_idx+1]))) / (beta + rho[ul_cur+6])
        z[ul_cur+7] = (-(lz[ul_cur+7] + l[ul_cur+7] + rho[ul_cur+7]*(u[ul_cur+7] - xbar[to_idx+1]))) / (beta + rho[ul_cur+7])
        ul_cur += 8
        xl_cur += 4
    end
end

function update_zu_generator_kernel(n::Int, gen_start::Int, u, xbar, z, l, rho, lz, beta)
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

function update_zu_branch_kernel(n::Int, line_start::Int, bus_start::Int, brBusIdx, u, xbar, z, l, rho, lz, beta)
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

function update_zv_kernel(n::Int, v, xbar, z, l, rho, lz, beta)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if tx <= n
        @inbounds begin
            z[tx] = (-(lz[tx] + l[tx] + rho[tx]*(v[tx] - xbar[tx]))) / (beta + rho[tx])
        end
    end

    return
end

function update_l_kernel(n::Int, l, z, lz, beta)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if tx <= n
        @inbounds begin
            l[tx] = -(lz[tx] + beta*z[tx])
        end
    end

    return
end

function compute_primal_residual_u(model::Model, rp_u, u, xbar, z)
    ngen = model.ngen
    nline = model.nline
    brBusIdx = model.brBusIdx
    gen_start, line_start, bus_start = model.gen_start, model.line_start, model.bus_start

    gen_end = gen_start + 2*ngen - 1
    rp_u[gen_start:gen_end] .= u[gen_start:gen_end] .- xbar[gen_start:gen_end] .+ z[gen_start:gen_end]

    ul_cur = line_start
    xl_cur = line_start
    for j=1:nline
        fr_idx = bus_start + 2*(brBusIdx[2*(j-1)+1]-1)
        to_idx = bus_start + 2*(brBusIdx[2*j]-1)

        rp_u[ul_cur:ul_cur+3] .= u[ul_cur:ul_cur+3] .- xbar[xl_cur:xl_cur+3] .+ z[ul_cur:ul_cur+3]
        rp_u[ul_cur+4] = u[ul_cur+4] - xbar[fr_idx] + z[ul_cur+4]
        rp_u[ul_cur+5] = u[ul_cur+5] - xbar[to_idx] + z[ul_cur+5]
        rp_u[ul_cur+6] = u[ul_cur+6] - xbar[fr_idx+1] + z[ul_cur+6]
        rp_u[ul_cur+7] = u[ul_cur+7] - xbar[to_idx+1] + z[ul_cur+7]

        ul_cur += 8
        xl_cur += 4
    end
end

function compute_primal_residual_u_generator_kernel(n::Int, gen_start::Int, rp, u, xbar, z)
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

function compute_primal_residual_u_branch_kernel(n::Int, line_start::Int, bus_start::Int, brBusIdx, rp, u, xbar, z)
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

function compute_primal_residual_v_kernel(n::Int, rp, v, xbar, z)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if tx <= n
        rp[tx] = v[tx] - xbar[tx] + z[tx]
    end

    return
end

function vector_difference(n::Int, c, a, b)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if tx <= n
        c[tx] = a[tx] - b[tx]
    end

    return
end

function update_lz_kernel(n::Int, max_limit::Float64, z, lz, beta)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if tx <= n
        lz[tx] = max(-max_limit, min(max_limit, lz[tx] + beta*z[tx]))
    end

    return
end

# TODO: Not used
function update_rho(rho, rp, rp_old, theta, gamma)
    for i=1:length(rho)
        if abs(rp[i]) > theta*abs(rp_old[i])
            rho[i] = min(gamma*rho[i], 1e24)
        end
    end
end

function set_rateA_kernel(nline::Int, param::CuDeviceArray{Float64,2}, rateA::CuDeviceArray{Float64,1})
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if tx <= nline
        param[29,tx] = (rateA[tx] == 0.0) ? Inf : rateA[tx]
    end

    return
end

function admm_restart_two_level(env::AdmmEnv, mod::Model; outer_iterlim=10, inner_iterlim=800, scale=1e-4)
    data, par = env.data, env.params
    sol = mod.solution

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

    nblk_gen = div(mod.ngen, 32, RoundUp)
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

    overall_time = @timed begin
    while outer < outer_iterlim
        outer += 1

        if !env.use_gpu
            z_outer .= z_curr
            z_prev_norm = norm(z_outer)
        else
            @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) copy_data_kernel(mod.nvar, z_outer, z_curr)
            z_prev_norm = CUDA.norm(z_curr)
            CUDA.synchronize()
        end

        inner = 0
        while inner < inner_iterlim
            inner += 1

            if !env.use_gpu
                z_prev .= z_curr
                rp_old .= rp

                tcpu = generator_kernel_two_level(mod, mod.baseMVA, u_curr, xbar_curr, zu_curr, lu_curr, rho_u)
                time_gen += tcpu.time

                #scale = min(scale, (2*1e4) / maximum(abs.(rho_u)))
                if !env.use_linelimit
                    tcpu = @timed auglag_it, tron_it = polar_kernel_two_level_cpu(mod.n, mod.nline, mod.line_start, mod.bus_start, scale,
                                                                                u_curr, xbar_curr, zu_curr, lu_curr, rho_u,
                                                                                shift_lines, mod.membuf, mod.YffR, mod.YffI, mod.YftR, mod.YftI,
                                                                                mod.YttR, mod.YttI, mod.YtfR, mod.YtfI, mod.FrBound, mod.ToBound, mod.brBusIdx)
                else
                    tcpu = @timed auglag_it, tron_it = auglag_linelimit_kernel_two_level_cpu(
                                                mod.n, mod.nline, mod.line_start, mod.bus_start,
                                                inner, par.max_auglag, par.mu_max, scale,
                                                u_curr, xbar_curr, zu_curr, lu_curr, rho_u,
                                                shift_lines, mod.membuf, mod.YffR, mod.YffI, mod.YftR, mod.YftI,
                                                mod.YttR, mod.YttI, mod.YtfR, mod.YtfI, mod.FrBound, mod.ToBound, mod.brBusIdx)
                end
                time_br += tcpu.time

                tcpu = @timed bus_kernel_two_level_cpu(mod.baseMVA, mod.nbus, mod.gen_start, mod.line_start, mod.bus_start,
                                                      mod.FrStart, mod.FrIdx, mod.ToStart, mod.ToIdx, mod.GenStart, mod.GenIdx,
                                                      mod.Pd, mod.Qd, v_curr, xbar_curr, zv_curr, lv_curr, rho_v, mod.YshR, mod.YshI)
                time_bus += tcpu.time

                update_xbar(mod, u_curr, v_curr, xbar_curr, zu_curr, zv_curr, lu_curr, lv_curr, rho_u, rho_v)

                update_zu(mod, u_curr, xbar_curr, zu_curr, lu_curr, rho_u, lz_u, beta)
                zv_curr .= (-(lz_v .+ lv_curr .+ rho_v.*(v_curr .- xbar_curr))) ./ (beta .+ rho_v)

                l_curr .= -(lz .+ beta.*z_curr)

                compute_primal_residual_u(mod, rp_u, u_curr, xbar_curr, zu_curr)
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
            else
                @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) copy_data_kernel(mod.nvar, z_prev, z_curr)
                @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) copy_data_kernel(mod.nvar, rp_old, rp)
                CUDA.synchronize()

                tgpu = generator_kernel_two_level(mod, mod.baseMVA, u_curr, xbar_curr, zu_curr, lu_curr, rho_u)
                # MPI routines to be implemented:
                #  - Broadcast v_curr and l_curr to GPUs.
                #  - Collect u_curr.
                #  - div(nblk_br / number of GPUs, RoundUp)

                time_gen += tgpu.time
                if !env.use_linelimit
                    tgpu = CUDA.@timed @cuda threads=32 blocks=nblk_br shmem=shmem_size polar_kernel_two_level(mod.n, mod.nline, mod.line_start, mod.bus_start, scale,
                                                              u_curr, xbar_curr, zu_curr, lu_curr, rho_u,
                                                              shift_lines, mod.membuf, mod.YffR, mod.YffI, mod.YftR, mod.YftI,
                                                              mod.YttR, mod.YttI, mod.YtfR, mod.YtfI, mod.FrBound, mod.ToBound, mod.brBusIdx)
                else
                    tgpu = CUDA.@timed @cuda threads=32 blocks=nblk_br shmem=shmem_size auglag_linelimit_kernel_two_level(
                                                              mod.n, mod.nline, mod.line_start, mod.bus_start,
                                                              inner, par.max_auglag, par.mu_max, scale,
                                                              u_curr, xbar_curr, zu_curr, lu_curr, rho_u,
                                                              shift_lines, mod.membuf, mod.YffR, mod.YffI, mod.YftR, mod.YftI,
                                                              mod.YttR, mod.YttI, mod.YtfR, mod.YtfI, mod.FrBound, mod.ToBound, mod.brBusIdx)
                end
                time_br += tgpu.time
                tgpu = CUDA.@timed @cuda threads=32 blocks=nblk_bus bus_kernel_two_level(mod.baseMVA, mod.nbus, mod.gen_start, mod.line_start, mod.bus_start,
                                                                     mod.FrStart, mod.FrIdx, mod.ToStart, mod.ToIdx, mod.GenStart, mod.GenIdx,
                                                                     mod.Pd, mod.Qd, v_curr, xbar_curr, zv_curr, lv_curr,
                                                                     rho_v, mod.YshR, mod.YshI)
                time_bus += tgpu.time

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
            end
        end

        if mismatch <= OUTER_TOL
            break
        end

        if !env.use_gpu
            lz .= max.(-par.MAX_MULTIPLIER, min.(par.MAX_MULTIPLIER, lz .+ (beta .* z_curr)))
        else
            CUDA.@sync @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) update_lz_kernel(mod.nvar, par.MAX_MULTIPLIER, z_curr, lz, beta)
        end

        if z_curr_norm > theta*z_prev_norm
            beta = min(c*beta, 1e24)
        end
    end
    end

    sol.objval = sum(data.generators[g].coeff[data.generators[g].n-2]*(mod.baseMVA*u_curr[mod.gen_start+2*(g-1)])^2 +
                data.generators[g].coeff[data.generators[g].n-1]*(mod.baseMVA*u_curr[mod.gen_start+2*(g-1)]) +
                data.generators[g].coeff[data.generators[g].n]
                for g in 1:mod.ngen)

    if par.verbose > 0
        # Test feasibility of global variable xbar:
        pg_err, qg_err = check_generator_bounds(mod, xbar_curr)
        vm_err = check_voltage_bounds(mod, xbar_curr)
        real_err, reactive_err = check_power_balance_violation(mod, xbar_curr)
        @printf(" ** Violations of global variable xbar\n")
        @printf("Real power generator bounds     = %.6e\n", pg_err)
        @printf("Reactive power generator bounds = %.6e\n", qg_err)
        @printf("Voltage bounds                  = %.6e\n", vm_err)
        @printf("Real power balance              = %.6e\n", real_err)
        @printf("Reaactive power balance         = %.6e\n", reactive_err)

        rateA_nviols, rateA_maxviol, rateC_nviols, rateC_maxviol = check_linelimit_violation(data, u_curr)
        @printf(" ** Line limit violations\n")
        @printf("RateA number of violations = %d (%d)\n", rateA_nviols, mod.nline)
        @printf("RateA maximum violation    = %.2f\n", rateA_maxviol)
        @printf("RateC number of violations = %d (%d)\n", rateC_nviols, mod.nline)
        @printf("RateC maximum violation    = %.2f\n", rateC_maxviol)

        @printf(" ** Time\n")
        @printf("Overall time    = %.2f\n", overall_time.time)
        @printf("Generator       = %.2f\n", time_gen)
        @printf("Branch          = %.2f\n", time_br)
        @printf("Bus             = %.2f\n", time_bus)
        @printf("Total(G+B+Br)   = %.2f\n", time_gen + time_br + time_bus)
        @printf("Objective value = %.6e\n", sol.objval)
    end

    return mismatch <= OUTER_TOL
end

function admm_rect_gpu_two_level(case::String;
    outer_iterlim=10, inner_iterlim=800, rho_pq=400.0, rho_va=40000.0, scale=1e-4,
    use_gpu=false, use_linelimit=false, outer_eps=2*1e-4, solve_pf=false, gpu_no=0, verbose=1)

    if use_gpu
        CUDA.device!(gpu_no)

        env = AdmmEnv{Float64, CuArray{Float64,1}, CuArray{Int,1}, CuArray{Float64,2}}(
            case, rho_pq, rho_va; use_gpu=use_gpu, use_linelimit=use_linelimit, use_twolevel=true,
            solve_pf=solve_pf, gpu_no=gpu_no, verbose=verbose,
        )
        env.params.outer_eps = outer_eps
        mod = Model{Float64, CuArray{Float64,1}, CuArray{Int,1}, CuArray{Float64,2}}(env)
        if use_linelimit
            # Set rateA in membuf.
            @cuda threads=64 blocks=(div(mod.nline-1, 64)+1) set_rateA_kernel(mod.nline, mod.membuf, mod.rateA)
        end
    else
        env = AdmmEnv{Float64, Array{Float64,1}, Array{Int,1}, Array{Float64,2}}(
            case, rho_pq, rho_va; use_gpu=use_gpu, use_linelimit=use_linelimit, use_twolevel=true,
            solve_pf=solve_pf, gpu_no=gpu_no, verbose=verbose,
        )
        env.params.outer_eps = outer_eps
        mod = Model{Float64, Array{Float64,1}, Array{Int,1}, Array{Float64,2}}(env)
        if use_linelimit
            mod.membuf[29,:] .= mod.rateA
        end
    end

    admm_restart_two_level(env, mod; outer_iterlim=outer_iterlim, inner_iterlim=inner_iterlim, scale=scale)
    return env, mod
end