function check_voltage_bounds_alternative(model::Model, v::Array{Float64,1})
    max_viol = 0.0

    for b=1:model.nbus
        if model.FrStart[b] < model.FrStart[b+1]
            l = model.FrIdx[model.FrStart[b]]
            wi = v[model.line_start + 8*(l-1) + 4]
        elseif model.ToStart[b] < model.ToStart[b+1]
            l = model.ToIdx[model.ToStart[b]]
            wi = v[model.line_start + 8*(l-1) + 5]
        else
            println("No lines connected to bus ", b)
        end

        err = max(max(0.0, wi - model.Vmax[b]^2), max(0.0, model.Vmin[b]^2 - wi))
        max_viol = (max_viol < err) ? err : max_viol
    end

    return max_viol
end

function check_power_balance_alternative(model::Model, u::Array{Float64,1}, v::Array{Float64,1})
    baseMVA = model.baseMVA
    nbus = model.nbus
    gen_start, line_start, YshR, YshI = model.gen_start, model.line_start, model.YshR, model.YshI

    max_viol_real = 0.0
    max_viol_reactive = 0.0
    for b=1:nbus
        real_err = 0.0
        reactive_err = 0.0
        for k=model.GenStart[b]:model.GenStart[b+1]-1
            g = model.GenIdx[k]
            real_err += u[gen_start + 2*(g-1)]
            reactive_err += u[gen_start + 2*(g-1)+1]
        end

        real_err -= (model.Pd[b] / baseMVA)
        reactive_err -= (model.Qd[b] / baseMVA)
        #real_err -= (data.buses[b].Pd / baseMVA)
        #reactive_err -= (data.buses[b].Qd / baseMVA)

        wi = 0
        for k=model.FrStart[b]:model.FrStart[b+1]-1
            l = model.FrIdx[k]
            real_err -= v[line_start + 8*(l-1)]
            reactive_err -= v[line_start + 8*(l-1) + 1]
            wi = v[line_start + 8*(l-1) + 4]
        end

        for k=model.ToStart[b]:model.ToStart[b+1]-1
            l = model.ToIdx[k]
            real_err -= v[line_start + 8*(l-1) + 2]
            reactive_err -= v[line_start + 8*(l-1) + 3]
            wi = v[line_start + 8*(l-1) + 5]
        end

        real_err -= YshR[b] * wi
        reactive_err += YshI[b] * wi

        max_viol_real = (max_viol_real < abs(real_err)) ? abs(real_err) : max_viol_real
        max_viol_reactive = (max_viol_reactive < abs(reactive_err)) ? abs(reactive_err) : max_viol_reactive
    end

    return max_viol_real, max_viol_reactive
end

function check_power_balance_alternative(model::Model, v::Array{Float64,1})
    baseMVA = model.baseMVA
    nbus = model.nbus
    gen_start, line_start, YshR, YshI = model.gen_start, model.line_start, model.YshR, model.YshI

    max_viol_real = 0.0
    max_viol_reactive = 0.0
    for b=1:nbus
        real_err = 0.0
        reactive_err = 0.0
        for k=model.GenStart[b]:model.GenStart[b+1]-1
            g = model.GenIdx[k]
            real_err += v[gen_start + 2*(g-1)]
            reactive_err += v[gen_start + 2*(g-1)+1]
        end

        real_err -= (model.Pd[b] / baseMVA)
        reactive_err -= (model.Qd[b] / baseMVA)
        #real_err -= (data.buses[b].Pd / baseMVA)
        #reactive_err -= (data.buses[b].Qd / baseMVA)

        wi = 0
        for k=model.FrStart[b]:model.FrStart[b+1]-1
            l = model.FrIdx[k]
            real_err -= v[line_start + 8*(l-1)]
            reactive_err -= v[line_start + 8*(l-1) + 1]
            wi = v[line_start + 8*(l-1) + 4]
        end

        for k=model.ToStart[b]:model.ToStart[b+1]-1
            l = model.ToIdx[k]
            real_err -= v[line_start + 8*(l-1) + 2]
            reactive_err -= v[line_start + 8*(l-1) + 3]
            wi = v[line_start + 8*(l-1) + 5]
        end

        real_err -= YshR[b] * wi
        reactive_err += YshI[b] * wi

        max_viol_real = (max_viol_real < abs(real_err)) ? abs(real_err) : max_viol_real
        max_viol_reactive = (max_viol_reactive < abs(reactive_err)) ? abs(reactive_err) : max_viol_reactive
    end

    return max_viol_real, max_viol_reactive
end