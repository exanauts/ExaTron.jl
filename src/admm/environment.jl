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

#=
struct ProxALGeneratorModel{TD} <: AbstractGeneratorModel
    model::AbstractOPFModel
    ngen::Int
    gen_start::Int
    pgmin::TD
    pgmax::TD
    qgmin::TD
    qgmax::TD
    c2::TD
    c1::TD
    c0::TD

    t_curr::Int      # current time period
    T::Int           # size of time horizon
    tau::Float64     # penalty for proximal term
    rho::Float64     # penalty for ramping equality
    pg_ref::TD       # proximal term
    pg_next::TD      # primal value for (t+1) time period
    l_next::TD       # dual (for ramping) value for (t-1) time period
    pg_prev::TD      # primal value for (t+1) time period
    l_prev::TD       # dual (for ramping) value for (t-1) time period
    s_curr::TD       # slack for ramping

    function ProxALGeneratorModel{TD}() where {TD}
        return new{TD}()
    end
end
=#

abstract type AbstractSolution{T,TD} end

"""
    SolutionOneLevel{T,TD}

This contains the solutions of ACOPF model instance, including the ADMM parameter rho.
"""
mutable struct SolutionOneLevel{T,TD} <: AbstractSolution{T,TD}
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

    function SolutionOneLevel{T,TD}(nvar::Int) where {T, TD<:AbstractArray{T}}
        sol = new{T,TD}(
            TD(undef, nvar),
            TD(undef, nvar),
            TD(undef, nvar),
            TD(undef, nvar),
            TD(undef, nvar),
            TD(undef, nvar),
            TD(undef, nvar),
            TD(undef, nvar),
            TD(undef, nvar),
            Inf,
        )

        sol.u_curr .= 0.0
        sol.v_curr .= 0.0
        sol.l_curr .= 0.0
        sol.u_prev .= 0.0
        sol.v_prev .= 0.0
        sol.l_prev .= 0.0
        sol.rho .= 0.0
        sol.rd .= 0.0
        sol.rp .= 0.0

        return sol
    end
end

"""
    SolutionTwoLevel{T,TD}

This contains the solutions of ACOPF model instance for two-level ADMM algorithm,
    including the ADMM parameter rho.
"""
mutable struct SolutionTwoLevel{T,TD} <: AbstractSolution{T,TD}
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

    function SolutionTwoLevel{T,TD}(nvar::Int, nvar_v::Int, nline::Int) where {T, TD<:AbstractArray{T}}
        sol = new{T,TD}(
            TD(undef, nvar),      # x_curr
            TD(undef, nvar_v),    # xbar_curr
            TD(undef, nvar),      # z_outer
            TD(undef, nvar),      # z_curr
            TD(undef, nvar),      # z_prev
            TD(undef, nvar),      # l_curr
            TD(undef, nvar),      # lz
            TD(undef, nvar),      # rho
            TD(undef, nvar),      # rp
            TD(undef, nvar),      # rd
            TD(undef, nvar),      # rp_old
            TD(undef, nvar),      # Ax_plus_By
            TD(undef, 2*nline),   # wRIij
            Inf,
        )
        sol.x_curr .= 0.0
        sol.xbar_curr .= 0.0
        sol.z_outer .= 0.0
        sol.z_curr .= 0.0
        sol.z_prev .= 0.0
        sol.l_curr .= 0.0
        sol.lz .= 0.0
        sol.rho .= 0.0
        sol.rp .= 0.0
        sol.rd .= 0.0
        sol.rp_old .= 0.0
        sol.Ax_plus_By .= 0.0
        sol.wRIij .= 0.0

        return sol
    end
end

abstract type AbstractOPFModel{T,TD,TI} end

"""
    Model{T,TD,TI}

This contains the parameters specific to ACOPF model instance.
"""
mutable struct Model{T,TD,TI} <: AbstractOPFModel{T,TD,TI}
    solution::AbstractSolution{T,TD}

    n::Int
    ngen::Int
    nline::Int
    nbus::Int
    nvar::Int

    gen_start::Int
    line_start::Int

    baseMVA::T
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

        model.baseMVA = data.baseMVA
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

        model.solution = ifelse(use_twolevel,
            SolutionTwoLevel{T,TD}(model.nvar, model.nvar_v, model.nline),
            SolutionOneLevel{T,TD}(model.nvar))

        return model
    end
end

abstract type AbstractAdmmEnv{T,TD,TI,TM} end

"""
    AdmmEnv{T,TD,TI}

This structure carries everything required to run ADMM from a given solution.
"""
mutable struct AdmmEnv{T,TD,TI,TM} <: AbstractAdmmEnv{T,TD,TI,TM}
    case::String
    data::OPFData
    use_gpu::Bool
    use_polar::Bool
    use_twolevel::Bool
    gpu_no::Int

    params::Parameters
    model::AbstractOPFModel{T,TD,TI}
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
        env.use_twolevel = use_twolevel

        env.params = Parameters()
        env.params.verbose = verbose

        env.model = Model{T,TD,TI}(env.data, use_gpu, use_polar, use_twolevel)
        ybus = Ybus{Array{T}}(computeAdmitances(
            env.data.lines, env.data.buses, env.data.baseMVA; VI=Array{Int}, VD=Array{T})...)

        init_solution!(env, env.model.solution, ybus, rho_pq, rho_va)

        env.membuf = TM(undef, (31, env.model.nline))
        fill!(env.membuf, 0.0)

        return env
    end
end

