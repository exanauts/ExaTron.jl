
using Test
using LinearAlgebra

CASE = joinpath(dirname(@__FILE__), "..", "data", "case9")

RAMP_AGC = [1.25, 1.5, 1.35]

LOADS = Dict(
    1 => Dict(
              "pd"=>[0.0, 0.0, 0.0, 0.0, 90.0, 0.0, 100.0, 0.0, 125.0],
              "qd"=>[0.0, 0.0, 0.0, 0.0, 30.0, 0.0, 35.0, 0.0, 50.0],
    ),
    2 => Dict(
              "pd"=>[0.0, 0.0, 0.0, 0.0, 97.3798938, 0.0, 108.199882, 0.0, 135.2498525],
              "qd"=>[0.0, 0.0, 0.0, 0.0, 32.4599646, 0.0, 37.8699587, 0.0, 54.099941],
    )
)

@testset "ProxAL wrapper" begin
    data = ExaTron.opf_loaddata(CASE)
    t, T = 2, 2
    rho_pq, rho_va = 400.0, 40000.0
    env = ExaTron.ProxALAdmmEnv(data, Array, t, T, rho_pq, rho_va; use_twolevel=true)
    @test isa(env, ExaTron.AdmmEnv)
    @test isa(env.model.gen_mod, ExaTron.ProxALGeneratorModel)

    ExaTron.set_proximal_term!(env, 0.1)
    ExaTron.set_penalty!(env, 0.1)
    ExaTron.set_upper_bound_slack!(env, 2 .* RAMP_AGC)

    ExaTron.set_active_load!(env, LOADS[t]["pd"])
    ExaTron.set_reactive_load!(env, LOADS[t]["qd"])

    ExaTron.admm_restart!(env)

    sol = env.solution
    @test sol.status == ExaTron.HAS_CONVERGED
end


