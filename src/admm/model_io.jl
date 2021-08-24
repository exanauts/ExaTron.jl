function fix_power_flow_parameters(
    data::OPFData,
    model::Model{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    for g=1:model.ngen
        pg_val = max(model.pgmin[g], min(model.pgmax[g], data.generators[g].Pg))
        model.pgmin[g] = model.pgmax[g] = pg_val
        model.c2[g] = model.c1[g] = model.c0[g] = 0.0
    end

    for b=1:model.nbus
        if data.buses[b].bustype in [2,3]
            vm_val = max(model.Vmin[b], min(model.Vmax[b], data.buses[b].Vm))
            model.Vmin[b] = model.Vmax[b] = vm_val
            for k=model.FrStart[b]:model.FrStart[b+1]-1
                l = model.FrIdx[k]
                model.FrBound[2*l-1] = model.FrBound[2*l] = vm_val
            end
            for k=model.ToStart[b]:model.ToStart[b+1]-1
                l = model.ToIdx[k]
                model.ToBound[2*l-1] = model.ToBound[2*l] = vm_val
            end
        end
    end

end