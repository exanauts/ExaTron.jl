

using Test
using LinearAlgebra

CASE = joinpath(dirname(@__FILE__), "..", "data", "case9")

@testset "OPFData" begin
    data = ExaTron.opf_loaddata(CASE)
    @test isa(data, ExaTron.OPFData)
end

@testset "AdmmEnv" begin
    rho_pq = 1.0
    rho_va = 1.0
    T = Float64
    VI = Vector{Int}
    VT = Vector{Float64}
    MT = Matrix{Float64}
    env = ExaTron.AdmmEnv{T,VT,VI,MT}(CASE, rho_pq, rho_va)

    @test env.case == CASE
    @test env.solution.status == ExaTron.INITIAL

    @testset "AdmmEnv.Model" begin
        model = env.model
        ngen = model.ngen
        @test length(model.pgmin) == length(model.pgmax) == length(model.qgmin) == length(model.qgmax) == ngen
        @test length(model.c2) == length(model.c1) == length(model.c0) == ngen

        nlines = model.nline
        @test length(model.FrBound) == length(model.ToBound) == 2 * nlines
        @test length(model.YshR) == length(model.YshI) == length(model.YffR) ==
              length(model.YffI) == length(model.YftR) == length(model.YftI) ==
              length(model.YttR) == length(model.YttI) == length(model.YtfR) ==
              length(model.YtfI) == nlines

        nbus = model.nbus
        @test length(model.Pd) == length(model.Qd) == nbus
    end
end

@testset "One-level ADMM algorithm" begin
    # NB: Need to run almost 2,000 iterations to reach convergence with this
    # set of parameters.
    env = ExaTron.admm_rect_gpu(CASE, Array; verbose=0, iterlim=2000, rho_pq=400.0, rho_va=40000.0)
    @test isa(env, ExaTron.AdmmEnv)

    model = env.model
    ngen = model.ngen

    par = env.params
    sol = env.solution

    # Check results
    pg = ExaTron.active_power_generation(env)
    qg = ExaTron.reactive_power_generation(env)

    @test sol.status == ExaTron.HAS_CONVERGED
    # Test with solution returned by PowerModels + Ipopt
    @test sol.objval ≈ 5296.6862 rtol=1e-4
    @test pg ≈ [0.897987, 1.34321, 0.941874] rtol=1e-4
    @test qg ≈ [0.1296564, 0.00031842, -0.226342] rtol=1e-2

    # Test restart API
    ExaTron.admm_restart!(env)
end

@testset "Two-level ADMM algorithm" begin
    # Two-level algorithm has around two times more coupling constraints
    # than the one-level algorithm, so it is expected to be less accurate
    # than the one level algorithm.
    env = ExaTron.admm_rect_gpu_two_level(
        CASE, Array;
        verbose=0, rho_pq=1000.0, rho_va=1000.0, inner_iterlim=2000, outer_iterlim=1500, outer_eps=1e-6
    )
    @test isa(env, ExaTron.AdmmEnv)

    model = env.model
    ngen = model.ngen

    par = env.params
    sol = env.solution

    # Check results
    pg = ExaTron.active_power_generation(env)
    qg = ExaTron.reactive_power_generation(env)

    @test sol.status == ExaTron.HAS_CONVERGED
    # # Test with solution returned by PowerModels + Ipopt
    @test sol.objval ≈ 5296.6862 rtol=1e-4
    @test pg ≈ [0.897987, 1.34321, 0.941874] rtol=1e-4
    @test qg ≈ [0.1296564, 0.00031842, -0.226342] atol=1e-2

    # Test restart API
    ExaTron.admm_restart_two_level!(env)
end

