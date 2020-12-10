
function build_problem(; n=10)
    Random.seed!(1)
    m = 0
    P = sparse(Diagonal(rand(n)) + 2.0 * sparse(I, n, n))
    q = randn(n)
    u =   1. * rand(n)
    l = - 100. * rand(n)
    Iz, Jz, vals = findnz(P)

    eval_f(x) = 0.5 * x' * P * x + q' * x

    function eval_g(x, g)
        fill!(g, 0)
        mul!(g, P, x)
        g .+= q
    end

    function eval_h(x, mode, rows, cols, obj_factor, lambda, values)
        if mode == :Structure
            for i in 1:nnz(P)
                rows[i] = Iz[i]
                cols[i] = Jz[i]
            end
        else
            copy!(values, vals)
        end
    end

    return ExaTron.createProblem(n, l, u, nnz(P), eval_f, eval_g, eval_h)
end

@testset "Problem definition" begin
    n = 1000
    obj♯ = -193.05853878066543
    prob = build_problem(; n=n)

    @testset "Problem definition" begin
        @test isa(prob, ExaTron.ExaTronProblem)
        @test length(prob.x) == length(prob.x_l) == length(prob.x_u) == n
        @test prob.status == :NotSolved
    end

    @testset "Tron: Julia" begin
        prob.x .= 0.5 .* (prob.x_l .+ prob.x_u)
        ExaTron.solveProblem(prob)
        @test prob.f ≈ obj♯ atol=1e-8
    end

    if ExaTron.has_c_library()
        @testset "Tron: Fortran" begin
            ExaTron.addOption(prob, "tron_code", :Fortran)
            prob.x .= 0.5 .* (prob.x_l .+ prob.x_u)
            ExaTron.solveProblem(prob)
            @test prob.f ≈ obj♯ atol=1e-8
        end
    end
end

