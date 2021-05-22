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
    verbose::Int

    # Two-Level ADMM
    outer_eps::Float64
    Kf::Int             # TODO: not used
    Kf_mean::Int        # TODO: not used
    MAX_MULTIPLIER::Float64
    DUAL_TOL::Float64

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
        par.verbose = 1
        par.outer_eps = 2*1e-4
        par.Kf = 100
        par.Kf_mean = 10
        par.MAX_MULTIPLIER = 1e12
        par.DUAL_TOL = 1e-8
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

    # Two-Level ADMM
    nvar_u::Int
    nvar_v::Int
    bus_start::Int # this is for varibles of type v.
    brBusIdx::TI

    function Model{T,TD,TI}(data::OPFData, use_gpu::Bool, use_polar::Bool, use_twolevel::Bool) where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}}
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

        # These are only for two-level ADMM.
        model.nvar_u = 2*model.ngen + 8*model.nline
        model.nvar_v = 2*model.ngen + 4*model.nline + 2*model.nbus
        model.bus_start = 2*model.ngen + 4*model.nline + 1
        if use_twolevel
            model.nvar = model.nvar_u + model.nvar_v
            model.brBusIdx = get_branch_bus_index(data; use_gpu=use_gpu)
        end

        return model
    end
end

abstract type AbstractSolution{T,TD} end

"""
    Solution{T,TD}

This contains the solutions of ACOPF model instance, including the ADMM parameter rho.
"""
mutable struct Solution{T,TD} <: AbstractSolution{T,TD}
    u_curr::TD
    v_curr::TD
    l_curr::TD
    u_prev::TD
    v_prev::TD
    l_prev::TD
    rho::TD
    rd::TD
    rp::TD
    objval::T

    function Solution{T,TD}(model::Model) where {T, TD<:AbstractArray{T}}
        return new{T,TD}(
            TD(undef, model.nvar),
            TD(undef, model.nvar),
            TD(undef, model.nvar),
            TD(undef, model.nvar),
            TD(undef, model.nvar),
            TD(undef, model.nvar),
            TD(undef, model.nvar),
            TD(undef, model.nvar),
            TD(undef, model.nvar),
            Inf,
        )
    end
end

"""
    Solution2{T,TD}

This contains the solutions of ACOPF model instance for two-level ADMM algorithm,
    including the ADMM parameter rho.
"""
mutable struct Solution2{T,TD} <: AbstractSolution{T,TD}
    x_curr::TD
    xbar_curr::TD
    z_outer::TD
    z_curr::TD
    z_prev::TD
    l_curr::TD
    lz::TD
    rho::TD
    rp::TD
    rd::TD
    rp_old::TD
    Ax_plus_By::TD
    wRIij::TD
    objval::T

    function Solution2{T,TD}(model::Model) where {T, TD<:AbstractArray{T}}
        return new{T,TD}(
            TD(undef, model.nvar),      # x_curr
            TD(undef, model.nvar_v),    # xbar_curr
            TD(undef, model.nvar),      # z_outer
            TD(undef, model.nvar),      # z_curr
            TD(undef, model.nvar),      # z_prev
            TD(undef, model.nvar),      # l_curr
            TD(undef, model.nvar),      # lz
            TD(undef, model.nvar),      # rho
            TD(undef, model.nvar),      # rp
            TD(undef, model.nvar),      # rd
            TD(undef, model.nvar),      # rp_old
            TD(undef, model.nvar),      # Ax_plus_By
            TD(undef, 2*model.nline),   # wRIij
            Inf,
        )
    end
end

"""
    AdmmEnv{T,TD,TI}

This structure carries everything required to run ADMM from a given solution.
"""
mutable struct AdmmEnv{T,TD,TI,TM}
    case::String
    data::OPFData
    use_gpu::Bool
    use_polar::Bool
    use_twolevel::Bool
    gpu_no::Int

    params::Parameters
    model::Model{T,TD,TI}
    solution::AbstractSolution{T,TD}

    membuf::TM # was param

    function AdmmEnv{T,TD,TI,TM}(
        case, rho_pq, rho_va; use_gpu=false, use_polar=true, use_twolevel=false, gpu_no=1, verbose=1,
    ) where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}, TM<:AbstractArray{T,2}}
        env = new{T,TD,TI,TM}()

        env.case = case
        env.data = opf_loaddata(env.case, Line(); VI=TI, VD=TD)
        env.use_gpu = use_gpu
        env.use_polar = use_polar
        env.gpu_no = gpu_no

        env.params = Parameters()
        env.params.verbose = verbose

        env.model = Model{T,TD,TI}(env.data, use_gpu, use_polar, use_twolevel)
        ybus = Ybus{Array{T}}(computeAdmitances(
            env.data.lines, env.data.buses, env.data.baseMVA; VI=Array{Int}, VD=Array{T})...)

        if !use_twolevel
            env.solution = Solution{T,TD}(env.model)
            init_solution!(env, ybus, rho_pq, rho_va)
        else
            env.solution = Solution2{T,TD}(env.model)
            init_values_two_level!(env, ybus, rho_pq, rho_va)
        end

        env.membuf = TM(undef, (31, env.model.nline))
        fill!(env.membuf, 0.0)

        return env
    end
end

