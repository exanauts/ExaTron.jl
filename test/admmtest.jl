

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
    env = ExaTron.admm_rect_gpu(CASE, Array; verbose=0, iterlim=2000, rho_pq=400.0, rho_va=40000.0)
    @test isa(env, ExaTron.AdmmEnv)

    model = env.model
    ngen = model.ngen

    par = env.params
    sol = env.solution

    # Check convergence
    primres = norm(sol.rp)
    dualres = norm(sol.rd)
    eps_pri = sqrt(length(sol.l_curr))*par.ABSTOL + par.RELTOL*max(norm(sol.u_curr), norm(-sol.v_curr))
    eps_dual = sqrt(length(sol.u_curr))*par.ABSTOL + par.RELTOL*norm(sol.l_curr)

    @test primres <= eps_pri && dualres <= eps_dual

    # Check results
    x♯ = sol.u_curr
    pg = x♯[1:2:2*ngen]
    qg = x♯[2:2:2*ngen]

    @test sol.status == ExaTron.HAS_CONVERGED
    # Test with solution returned by PowerModels + Ipopt
    @test pg ≈ [0.897987, 1.34321, 0.941874] atol=1e-3
    @test qg ≈ [0.1296564, 0.00031842, -0.226342] atol=1e-3
    @test sol.objval ≈ 5296.6862 rtol=1e-4

    # Test restart API
    ExaTron.admm_restart!(env)
end

