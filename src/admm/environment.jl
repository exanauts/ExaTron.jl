"""
    Parameters

This contains the parameters used in ADMM algorithm.
"""
mutable struct Parameters
    mu_max::Float64 # Augmented Lagrangian
    max_auglag::Int # Augmented Lagrangian
    ABSTOL::Float64
    RELTOL::Float64

    rho_max::Float64    # TODO: not used
    rho_min_pq::Float64 # TODO: not used
    rho_min_w::Float64  # TODO: not used
    eps_rp::Float64     # TODO: not used
    eps_rp_min::Float64 # TODO: not used
    rt_inc::Float64     # TODO: not used
    rt_dec::Float64     # TODO: not used
    eta::Float64        # TODO: not used

    function Parameters()
        par = new()
        par.mu_max = 1e8
        par.rho_max = 1e6
        par.rho_min_pq = 5.0
        par.rho_min_w = 5.0
        par.eps_rp = 1e-4
        par.eps_rp_min = 1e-5
        par.rt_inc = 2.0
        par.rt_dec = 2.0
        par.eta = 0.99
        par.max_auglag = 50
        par.ABSTOL = 1e-6
        par.RELTOL = 1e-5
        return par
    end
end

"""
    Model{T,TD,TI}

This contains the parameters specific to ACOPF model instance.
"""
mutable struct Model{T,TD,TI}
    n::Int
    ngen::Int
    nline::Int
    nbus::Int
    nvar::Int
    
    gen_start::Int
    line_start::Int

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

    function Model{T,TD,TI}(data::OPFData, use_gpu::Bool, use_polar::Bool) where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}}
        model = new{T,TD,TI}()
        
        model.n = (use_polar == true) ? 4 : 10
        model.ngen = length(data.generators)
        model.nline = length(data.lines)
        model.nbus = length(data.buses)
        model.nvar = 2*model.ngen + 8*model.nline
        model.gen_start = 1
        model.line_start = 2*model.ngen + 1
        model.pgmin, model.pgmax, model.qgmin, model.qgmax, model.c2, model.c1, model.c0 = get_generator_data(data; use_gpu=use_gpu)
        model.YshR, model.YshI, model.YffR, model.YffI, model.YftR, model.YftI, model.YttR, model.YttI, model.YtfR, model.YtfI, model.FrBound, model.ToBound = get_branch_data(data; use_gpu=use_gpu)
        model.FrStart, model.FrIdx, model.ToStart, model.ToIdx, model.GenStart, model.GenIdx, model.Pd, model.Qd = get_bus_data(data; use_gpu=use_gpu)

        return model
    end
end

"""
    Solution{T,TD}

This contains the solutions of ACOPF model instance, including the ADMM parameter rho.
"""
mutable struct Solution{T,TD}
    u_curr::TD
    v_curr::TD
    l_curr::TD
    u_prev::TD
    v_prev::TD
    l_prev::TD
    rho::TD
    rd::TD
    rp::TD

    function Solution{T,TD}(data::OPFData, model::Model, rho_pq, rho_va) where {T, TD<:AbstractArray{T}}
        sol = new{T,TD}()
        u_curr = zeros(model.nvar)
        v_curr = zeros(model.nvar)
        l_curr = zeros(model.nvar)
        u_prev = zeros(model.nvar)
        v_prev = zeros(model.nvar)
        l_prev = zeros(model.nvar)
        rho = zeros(model.nvar)
        rd = zeros(model.nvar)
        rp = zeros(model.nvar)
        wRIij = zeros(2*model.nline) # TODO: Not used

        ybus = Ybus{Array{Float64}}(computeAdmitances(
            data.lines, data.buses, data.baseMVA; VI=Array{Int}, VD=Array{Float64})...)
    
        init_values(data, ybus, model.gen_start, model.line_start,
                    rho_pq, rho_va, u_curr, v_curr, l_curr, rho, wRIij)

        sol.u_curr = TD(undef, model.nvar)
        sol.v_curr = TD(undef, model.nvar)
        sol.l_curr = TD(undef, model.nvar)
        sol.u_prev = TD(undef, model.nvar)
        sol.v_prev = TD(undef, model.nvar)
        sol.l_prev = TD(undef, model.nvar)
        sol.rho = TD(undef, model.nvar)
        sol.rd = TD(undef, model.nvar)
        sol.rp = TD(undef, model.nvar)

        copyto!(sol.u_curr, u_curr)
        copyto!(sol.v_curr, v_curr)
        copyto!(sol.l_curr, l_curr)
        copyto!(sol.rho, rho)
        return sol
    end
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

    params::Parameters
    model::Model{T,TD,TI}
    solution::Solution{T,TD}

    membuf::TD # was param

    function AdmmEnv{T,TD,TI}(case, rho_pq, rho_va; use_gpu=false, use_polar=false, gpu_no=1) where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}}
        env = new{T,TD,TI}()

        env.case = case
        env.data = opf_loaddata(env.case)
        env.use_gpu = use_gpu
        env.use_polar = use_polar
        env.gpu_no = gpu_no

        env.params = Parameters()
        env.model = Model{T,TD,TI}(env.data, use_gpu, use_polar)
        env.solution = Solution{T,TD}(env.data, env.model, rho_pq, rho_va)

        env.membuf = TD(undef, (31, env.model.nline))
        fill!(env.membuf, 0.0)

        return env
    end
end
