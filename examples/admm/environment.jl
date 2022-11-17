@enum(Status::Int,
    INITIAL = 0,
    HAS_CONVERGED = 1,
    MAXIMUM_ITERATIONS = 2,
)

"""
    Parameters

This contains the parameters used in ADMM algorithm.
"""
mutable struct Parameters
    mu_max::Float64 # Augmented Lagrangian
    max_auglag::Int # Augmented Lagrangian
    abstol::Float64
    reltol::Float64

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
    rho_sigma::Float64  # Penalty for bus

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
        par.abstol = 1e-6
        par.reltol = 1e-5
        par.verbose = 1
        par.outer_eps = 2*1e-4
        par.Kf = 100
        par.Kf_mean = 10
        par.MAX_MULTIPLIER = 1e12
        par.DUAL_TOL = 1e-8
        par.rho_sigma = 1e8
        return par
    end
end

abstract type AbstractGeneratorModel end

struct GeneratorModel{TD} <: AbstractGeneratorModel
    ngen::Int
    gen_start::Int
    pgmin::TD
    pgmax::TD
    qgmin::TD
    qgmax::TD
    c2::TD
    c1::TD
    c0::TD
end

"""
    Model{T,TD,TI}

This contains the parameters specific to ACOPF model instance.
"""
mutable struct Model{T,TD,TI}
    n::Int
    nline::Int
    nbus::Int
    nvar::Int

    line_start::Int

    gen_mod::AbstractGeneratorModel
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

    function Model{T,TD,TI}(data::OPFData, device) where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}}
        model = new{T,TD,TI}()

        ngen = length(data.generators)
        gen_start = 1
        pgmin, pgmax, qgmin, qgmax, c2, c1, c0 = get_generator_data(data, device)
        model.gen_mod = GeneratorModel{TD}(ngen, gen_start, pgmin, pgmax, qgmin, qgmax, c2, c1, c0)

        model.n = 4
        model.nline = length(data.lines)
        model.nbus = length(data.buses)
        model.nvar = 2*ngen + 8*model.nline
        model.line_start = 2*ngen + 1
        model.YshR, model.YshI, model.YffR, model.YffI, model.YftR, model.YftI, model.YttR, model.YttI, model.YtfR, model.YtfI, model.FrBound, model.ToBound = get_branch_data(data, device)
        model.FrStart, model.FrIdx, model.ToStart, model.ToIdx, model.GenStart, model.GenIdx, model.Pd, model.Qd = get_bus_data(data, device)

        # These are only for two-level ADMM.
        model.nvar_u = 2*ngen + 8*model.nline
        model.nvar_v = 2*ngen + 4*model.nline + 2*model.nbus
        model.bus_start = 2*ngen + 4*model.nline + 1
        return model
    end
end

abstract type AbstractSolution{T,TD} end

"""
    SolutionOneLevel{T,TD}

This contains the solutions of ACOPF model instance, including the ADMM parameter rho.
"""
mutable struct SolutionOneLevel{T,TD} <: AbstractSolution{T,TD}
    status::Status
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

    function SolutionOneLevel{T,TD}(model::Model) where {T, TD<:AbstractArray{T}}
        sol = new{T,TD}(
            INITIAL,
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

function active_power_generation(model::Model, sol::SolutionOneLevel)
    ngen = model.gen_mod.ngen
    return sol.u_curr[1:2:2*ngen]
end
function reactive_power_generation(model::Model, sol::SolutionOneLevel)
    ngen = model.gen_mod.ngen
    return sol.u_curr[2:2:2*ngen]
end
function voltage_magnitude(model::Model, sol::SolutionOneLevel)
    nbus = model.nbus
    return zeros(nbus)
end
function voltage_angle(model::Model, sol::SolutionOneLevel)
    nbus = model.nbus
    return zeros(nbus)
end

function set_active_power_generation!(model::Model, sol::SolutionOneLevel, val)
    ngen = model.gen_mod.ngen
    for g in 1:ngen
        pg_idx = model.gen_mod.gen_start + 2 * (g-1)
        @inbounds sol.v_curr[pg_idx] = val[g]
    end
end
function set_reactive_power_generation!(model::Model, sol::SolutionOneLevel, val)
    ngen = model.gen_mod.ngen
    for g in 1:ngen
        pg_idx = model.gen_mod.gen_start + 2 * (g-1)
        @inbounds sol.v_curr[pg_idx+1] = val[g]
    end
end

"""
    AdmmEnv{T,TD,TI}

This structure carries everything required to run ADMM from a given solution.
"""
mutable struct AdmmEnv{T,TD,TI,TM}
    case::String
    data::OPFData
    allow_infeas::Bool

    params::Parameters
    model::Model{T,TD,TI}
    solution::AbstractSolution{T,TD}

    membuf::TM # was param
    device

    function AdmmEnv{T,TD,TI,TM}(
        opfdata, rho_pq, rho_va; verbose=1,
        allow_infeas=false, rho_sigma=1e8,
        outer_eps=2e-4, device=KA.CPU()
    ) where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}, TM<:AbstractArray{T,2}}
        env = new{T,TD,TI,TM}()

        env.data = opfdata
        env.device = device
        env.allow_infeas = allow_infeas

        env.params = Parameters()
        env.params.verbose = verbose
        env.params.outer_eps = outer_eps
        env.params.rho_sigma = rho_sigma

        env.model = Model{T,TD,TI}(env.data, device)
        ybus = Ybus{Array{T}}(computeAdmitances(
            env.data.lines, env.data.buses, env.data.baseMVA, device; VI=Array{Int}, VD=Array{T})...)

        env.solution = SolutionOneLevel{T,TD}(env.model)
        init_solution!(env, env.solution, ybus, rho_pq, rho_va)

        env.membuf = TM(undef, (31, env.model.nline))
        fill!(env.membuf, 0.0)

        return env
    end
end

function AdmmEnv(case::String, device, rho_pq, rho_va; options...)
    opfdata = opf_loaddata(case)
    env = AdmmEnv(opfdata, device, rho_pq, rho_va; options...)
    env.case = case
    return env
end

AdmmEnv(opfdata::OPFData, unknown, rho_pq, rho_va; options...) = error("Unknown device $(typeof(unknown))")

AdmmEnv(opfdata::OPFData, device::KA.CPU, rho_pq, rho_va; options...) = AdmmEnv{Float64, Array{Float64, 1}, Array{Int, 1}, Array{Float64, 2}}(
        opfdata, rho_pq, rho_va; device=device, options...
    )

AdmmEnv(opfdata::OPFData, device::CUDADevice, rho_pq, rho_va; options...) = AdmmEnv{Float64, CuArray{Float64, 1}, CuArray{Int, 1}, CuArray{Float64, 2}}(
        opfdata, rho_pq, rho_va; device=device, options...
    )

AdmmEnv(opfdata::OPFData, device::ROCDevice, rho_pq, rho_va; options...) = AdmmEnv{Float64, ROCArray{Float64, 1}, ROCArray{Int, 1}, ROCArray{Float64, 2}}(
        opfdata, rho_pq, rho_va; device=device, options...
    )


# Getters / setters
active_power_generation(env::AdmmEnv) = active_power_generation(env.model, env.solution)
reactive_power_generation(env::AdmmEnv) = reactive_power_generation(env.model, env.solution)

function set_active_load!(env::AdmmEnv, pd::AbstractArray)
    @assert length(pd) == env.model.nbus
    copyto!(env.model.Pd, pd)
    return
end

function set_reactive_load!(env::AdmmEnv, qd::AbstractArray)
    @assert length(qd) == env.model.nbus
    copyto!(env.model.Qd, qd)
    return
end

set_active_power_generation!(env::AdmmEnv, val) = set_active_power_generation!(env.model, env.solution, val)
set_reactive_power_generation!(env::AdmmEnv, val) = set_reactive_power_generation!(env.model, env.solution, val)
set_voltage_magnitude!(env::AdmmEnv, val) = set_voltage_magnitude!(env.model, env.solution, val)
set_voltage_angle!(env::AdmmEnv, val) = set_voltage_angle!(env.model, env.solution, val)

