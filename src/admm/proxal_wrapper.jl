
mutable struct ProxALGeneratorModel{TD} <: AbstractGeneratorModel
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
    smin::TD         # slack's lower bound
    smax::TD         # slack's upper bound

    Q_ref::TD
    c_ref::TD

    Q::TD
    c::TD
end
function ProxALGeneratorModel(modelgen::GeneratorModel{TD}, t::Int, T::Int) where TD
    pg_ref  = TD(undef, modelgen.ngen)  ; fill!(pg_ref, 0.0)
    pg_next = TD(undef, modelgen.ngen)  ; fill!(pg_next, 0.0)
    l_next  = TD(undef, modelgen.ngen)  ; fill!(l_next, 0.0)
    pg_prev = TD(undef, modelgen.ngen)  ; fill!(pg_prev, 0.0)
    l_prev  = TD(undef, modelgen.ngen)  ; fill!(l_prev, 0.0)
    s_curr  = TD(undef, modelgen.ngen)  ; fill!(s_curr, 0.0)
    s_min   = TD(undef, modelgen.ngen)  ; fill!(s_min, 0.0)
    s_max   = TD(undef, modelgen.ngen)  ; fill!(s_max, 0.0)
    Q       = TD(undef, modelgen.ngen*4); fill!(Q, 0.0)
    c       = TD(undef, modelgen.ngen*2); fill!(c, 0.0)
    Q_ref   = TD(undef, modelgen.ngen*4); fill!(Q_ref, 0.0)
    c_ref   = TD(undef, modelgen.ngen*2); fill!(c_ref, 0.0)
    return ProxALGeneratorModel{TD}(
        modelgen.ngen, modelgen.gen_start,
        modelgen.pgmin, modelgen.pgmax, modelgen.qgmin, modelgen.qgmax, modelgen.c2, modelgen.c1, modelgen.c0,
        t, T, 0.1, 0.1, pg_ref, pg_next, l_next, pg_prev, l_prev, s_curr, s_min, s_max, Q_ref, c_ref, Q, c
    )
end

function ProxALAdmmEnv(opfdata::OPFData, use_gpu::Bool, t, T, rho_pq, rho_va; options...)
    env = AdmmEnv(opfdata, use_gpu, rho_pq, rho_va; options...)
    model = env.model
    # Replace generator's model by ProxAL model
    env.model.gen_mod = ProxALGeneratorModel(env.model.gen_mod, t, T)
    return env
end

# GETTERS
slack_values(env::AdmmEnv) = env.model.gen_mod.s_curr

# SETTERS
macro define_setter_array(function_name, attribute)
    fname = Symbol(function_name)
    quote
        function $(esc(fname))(env::AdmmEnv, values::AbstractVector)
            model = env.model
            @assert isa(model.gen_mod, ProxALGeneratorModel)
            copyto!(model.gen_mod.$attribute, values)
            return
        end
    end
end

macro define_setter_value(function_name, attribute)
    fname = Symbol(function_name)
    quote
        function $(esc(fname))(env::AdmmEnv, value::AbstractFloat)
            model = env.model
            @assert isa(model.gen_mod, ProxALGeneratorModel)
            model.gen_mod.$attribute = value
            return
        end
    end
end

@define_setter_array set_lower_bound_slack! smin
@define_setter_array set_upper_bound_slack! smax
@define_setter_array set_slack! s_curr
@define_setter_array set_multiplier_last! l_prev
@define_setter_array set_multiplier_next! l_next
@define_setter_array set_proximal_last! pg_prev
@define_setter_array set_proximal_next! pg_next
@define_setter_array set_proximal_ref! pg_ref

@define_setter_value set_proximal_term! tau
@define_setter_value set_penalty! rho
