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

    outer_iterlim::Int
    inner_iterlim::Int
    scale::Float64

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

        par.outer_iterlim = 20
        par.inner_iterlim = 1000
        par.scale = 1e-4

        return par
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
    load::Load{TM}
    initial_rho_pq::Float64
    initial_rho_va::Float64
    horizon_length::Int
    use_gpu::Bool
    use_linelimit::Bool
    use_twolevel::Bool
    solve_pf::Bool
    gpu_no::Int

    params::Parameters
#    model::AbstractOPFModel{T,TD,TI}
#    membuf::TM # was param

    function AdmmEnv{T,TD,TI,TM}(
        case::String, rho_pq::Float64, rho_va::Float64;
        use_gpu=false, use_linelimit=false, use_twolevel=false,
        solve_pf=false, gpu_no::Int=1, verbose::Int=1,
        horizon_length=1, load_prefix::String=""
    ) where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}, TM<:AbstractArray{T,2}}
        env = new{T,TD,TI,TM}()

        env.case = case
        env.data = opf_loaddata(env.case, Line(); VI=TI, VD=TD)
        env.initial_rho_pq = rho_pq
        env.initial_rho_va = rho_va
        env.use_gpu = use_gpu
        env.use_linelimit = use_linelimit
        env.gpu_no = gpu_no
        env.use_twolevel = use_twolevel
        env.solve_pf = solve_pf

        env.params = Parameters()
        env.params.verbose = verbose

        env.horizon_length = horizon_length
        if !isempty(load_prefix)
            env.load = get_load(load_prefix; use_gpu=use_gpu)
            @assert size(env.load.pd) == size(env.load.qd)
            @assert size(env.load.pd,2) >= horizon_length && size(env.load.qd,2) >= horizon_length
        end

#=
        ybus = Ybus{Array{T}}(computeAdmitances(
            env.data.lines, env.data.buses, env.data.baseMVA; VI=Array{Int}, VD=Array{T})...)

        if !isempty(multiperiod_load)
            @assert time_index >= 1
            env.load = get_load(multiperiod_load; use_gpu=use_gpu)
            env.model = ModelWithRamping{T,TD,TI}(env.data, time_index,
                                env.load.pd[:,time_index], env.load.qd[:,time_index])
            env.is_multiperiod = true
            init_solution!(env, env.model.single_period.solution, ybus, rho_pq, rho_va)
            env.membuf = TM(undef, (31, env.model.single_period.nline))
            fill!(env.membuf, 0.0)
        else
            env.model = Model{T,TD,TI}(env.data, use_gpu, linelimit, use_twolevel)
            env.is_multiperiod = false
            init_solution!(env, env.model.solution, ybus, rho_pq, rho_va)
            env.membuf = TM(undef, (31, env.model.nline))
            fill!(env.membuf, 0.0)
        end
=#

        return env
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

        fill!(sol, 0.0)

        return sol
    end
end

function Base.fill!(sol::SolutionTwoLevel, val)
    fill!(sol.x_curr, val)
    fill!(sol.xbar_curr, val)
    fill!(sol.z_outer, val)
    fill!(sol.z_curr, val)
    fill!(sol.z_prev, val)
    fill!(sol.l_curr, val)
    fill!(sol.lz, val)
    fill!(sol.rho, val)
    fill!(sol.rp, val)
    fill!(sol.rd, val)
    fill!(sol.rp_old, val)
    fill!(sol.Ax_plus_By, val)
    fill!(sol.wRIij, val)
end

abstract type AbstractOPFModel{T,TD,TI,TM} end

"""
    Model{T,TD,TI}

This contains the parameters specific to ACOPF model instance.
"""
mutable struct Model{T,TD,TI,TM} <: AbstractOPFModel{T,TD,TI,TM}
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
    rateA::TD
    FrStart::TI
    FrIdx::TI
    ToStart::TI
    ToIdx::TI
    GenStart::TI
    GenIdx::TI
    Pd::TD
    Qd::TD
    Vmin::TD
    Vmax::TD

    membuf::TM

    # Two-Level ADMM
    nvar_u::Int
    nvar_v::Int
    bus_start::Int # this is for varibles of type v.
    brBusIdx::TI

    function Model{T,TD,TI,TM}(env::AdmmEnv{T,TD,TI,TM}) where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}, TM<:AbstractArray{T,2}}
        model = new{T,TD,TI,TM}()

        model.baseMVA = env.data.baseMVA
        model.n = (env.use_linelimit == true) ? 6 : 4
        model.ngen = length(env.data.generators)
        model.nline = length(env.data.lines)
        model.nbus = length(env.data.buses)
        model.nvar = 2*model.ngen + 8*model.nline
        model.gen_start = 1
        model.line_start = 2*model.ngen + 1
        model.pgmin, model.pgmax, model.qgmin, model.qgmax, model.c2, model.c1, model.c0 = get_generator_data(env.data; use_gpu=env.use_gpu)
        model.YshR, model.YshI, model.YffR, model.YffI, model.YftR, model.YftI, model.YttR, model.YttI, model.YtfR, model.YtfI, model.FrBound, model.ToBound, model.rateA = get_branch_data(env.data; use_gpu=env.use_gpu)
        model.FrStart, model.FrIdx, model.ToStart, model.ToIdx, model.GenStart, model.GenIdx, model.Pd, model.Qd, model.Vmin, model.Vmax = get_bus_data(env.data; use_gpu=env.use_gpu)
        model.brBusIdx = get_branch_bus_index(env.data; use_gpu=env.use_gpu)

        if env.solve_pf
            fix_power_flow_parameters(env.data, model)
        end

        # These are only for two-level ADMM.
        model.nvar_u = 2*model.ngen + 8*model.nline
        model.nvar_v = 2*model.ngen + 4*model.nline + 2*model.nbus
        model.bus_start = 2*model.ngen + 4*model.nline + 1
        if env.use_twolevel
            model.nvar = model.nvar_u + model.nvar_v
        end

        model.solution = ifelse(env.use_twolevel,
            SolutionTwoLevel{T,TD}(model.nvar, model.nvar_v, model.nline),
            SolutionOneLevel{T,TD}(model.nvar))
        init_solution!(model, model.solution, env.initial_rho_pq, env.initial_rho_va)

        model.membuf = TM(undef, (31, model.nline))
        fill!(model.membuf, 0.0)

        return model
    end
end

mutable struct SolutionRamping{T,TD} <: AbstractSolution{T,TD}
    x_curr::TD        # (phat_{t,g}, s_{t,g}): 2*|G|
    xbar_curr::TD     # (ptilde_{t,g}): |G|
    xbar_tm1_curr::TD # (ptilde_{t-1,g}): |G|
    z_outer::TD       # (phat_{t,g}, s_{t,g}, ptilde_{t,g}): 3*|G|
    z_curr::TD        # (phat_{t,g}, s_{t,g}, ptilde_{t,g}): 3*|G|
    z_prev::TD        # (phat_{t,g}, s_{t,g}, ptilde_{t,g}): 3*|G|
    l_curr::TD        # (phat_{t,g}, s_{t,g}, ptilde_{t,g}): 3*|G|
    lz::TD            # (phat_{t,g}, s_{t,g}, ptilde_{t,g}): 3*|G|
    rho::TD           # (phat_{t,g}, s_{t,g}, ptilde_{t,g}): 3*|G|
    rp::TD            # (phat_{t,g}, s_{t,g}, ptilde_{t,g}): 3*|G|
    rd::TD            # (phat_{t,g}, s_{t,g}, ptilde_{t,g}): 3*|G|
    Ax_plus_By::TD    # (phat_{t,g}, s_{t,g}, ptilde_{t,g}): 3*|G|

    # Views for a shortcut to the above variables.
    # All of these views point to continuous memory space.

    Z_HAT::AbstractArray{T}
    Z_S::AbstractArray{T}
    Z_TILDE::AbstractArray{T}
    L_HAT::AbstractArray{T}
    L_S::AbstractArray{T}
    L_TILDE::AbstractArray{T}
    RHO_HAT::AbstractArray{T}
    RHO_S::AbstractArray{T}
    RHO_TILDE::AbstractArray{T}

    function SolutionRamping{T,TD}(ngen::Int) where {T,TD<:AbstractArray{T}}
        sol = new{T,TD}()
        sol.x_curr = TD(undef, 2*ngen)
        sol.xbar_curr = TD(undef, ngen)
        sol.xbar_tm1_curr = TD(undef, ngen)
        sol.z_outer = TD(undef, 3*ngen)
        sol.z_curr = TD(undef, 3*ngen)
        sol.z_prev = TD(undef, 3*ngen)
        sol.l_curr = TD(undef, 3*ngen)
        sol.lz = TD(undef, 3*ngen)
        sol.rho = TD(undef, 3*ngen)
        sol.rp = TD(undef, 3*ngen)
        sol.rd = TD(undef, 3*ngen)
        sol.Ax_plus_By = TD(undef, 3*ngen)

        sol.Z_HAT = view(sol.z_curr, 1:ngen)
        sol.Z_S = view(sol.z_curr, ngen+1:2*ngen)
        sol.Z_TILDE = view(sol.z_curr, 2*ngen+1:3*ngen)
        sol.L_HAT = view(sol.l_curr, 1:ngen)
        sol.L_S = view(sol.l_curr, ngen+1:2*ngen)
        sol.L_TILDE = view(sol.l_curr, 2*ngen+1:3*ngen)
        sol.RHO_HAT = view(sol.rho, 1:ngen)
        sol.RHO_S = view(sol.rho, ngen+1:2*ngen)
        sol.RHO_TILDE = view(sol.rho, 2*ngen+1:3*ngen)

        fill!(sol, 0.0)

        return sol
    end
end

function Base.fill!(sol::SolutionRamping, val)
    fill!(sol.x_curr, val)
    fill!(sol.xbar_curr, val)
    fill!(sol.xbar_tm1_curr, val)
    fill!(sol.z_outer, val)
    fill!(sol.z_curr, val)
    fill!(sol.z_prev, val)
    fill!(sol.l_curr, val)
    fill!(sol.lz, val)
    fill!(sol.rho, val)
    fill!(sol.rp, val)
    fill!(sol.rd, val)
    fill!(sol.Ax_plus_By, val)
end

mutable struct ModelWithRamping{T,TD,TI,TM} <: AbstractOPFModel{T,TD,TI,TM}
    inner::Model{T,TD,TI,TM}                  # single-period model without ramping
    ramping_solution::AbstractSolution{T,TD}  # ramping-specific solution
    time_index::Int                           # time-period index
    gen_membuf::TM                            # buffer for generator kernel
    ramp_rate::TD                             # ramp rate of generators

    function ModelWithRamping{T,TD,TI,TM}(
        env::AdmmEnv{T,TD,TI,TM}, time_index::Int; ramp_ratio=0.2) where {T,TD<:AbstractArray{T},TI<:AbstractArray{Int},TM<:AbstractArray{T,2}}
        model = new{T,TD,TI,TM}()
        model.inner = Model{T,TD,TI,TM}(env)
        model.inner.Pd = env.load.pd[:,time_index]
        model.inner.Qd = env.load.qd[:,time_index]
        model.time_index = time_index

        if env.use_gpu
            model.ramp_rate = TD(undef, model.inner.ngen)
            model.ramp_rate .= ramp_ratio .* model.inner.pgmax
        else
            model.ramp_rate = [ramp_ratio*model.inner.pgmax[g] for g=1:model.inner.ngen]
        end

        model.ramping_solution = SolutionRamping{T,TD}(model.inner.ngen)
        model.gen_membuf = TM(undef, (12, model.inner.ngen))
        fill!(model.gen_membuf, 0.0)
        init_solution!(model, env.initial_rho_pq)

        return model
    end
end
