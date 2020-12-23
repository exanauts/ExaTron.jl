function build_problem(; n=10)
    Random.seed!(1)
    m = 0
    P = sparse(Diagonal(rand(n)) + 2.0 * sparse(I, n, n))
    q = randn(n)
    u =   1. * rand(n)
    l = - 100. * rand(n)
    Iz, Jz, vals = findnz(P)

    eval_f(x) = 0.5 * dot(x, P, x) + dot(q, x)

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

@testset "PosDef QP: IncompleteCholesky" begin
    n = 1000
    obj♯ = -193.05853878066543
    iter♯ = 3
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
        @test prob.minor_iter == iter♯
    end

    if ExaTron.has_c_library()
        @testset "Tron: Fortran" begin
            ExaTron.setOption(prob, "tron_code", :Fortran)
            prob.x .= 0.5 .* (prob.x_l .+ prob.x_u)
            ExaTron.solveProblem(prob)
            @test prob.f ≈ obj♯ atol=1e-8
            @test prob.minor_iter == iter♯
        end
    end

end

@testset "PosDef QP: EyePreconditionner" begin
    n = 1000
    obj♯ = -193.05853878066543
    prob = build_problem(; n=n)
    @testset "Tron: Julia" begin
        prob.x .= 0.5 .* (prob.x_l .+ prob.x_u)
        prob.precond = ExaTron.EyePreconditionner()
        ExaTron.solveProblem(prob)
        @test prob.f ≈ obj♯ atol=1e-8
    end
    # Fortran backend supports only IncompleteCholesky
    if ExaTron.has_c_library()
        ExaTron.setOption(prob, "tron_code", :Fortran)
        @test_throws Exception ExaTron.solveProblem(prob)
    end
end

@testset "1-d QP" begin
    # Solve a simple QP problem: min 0.5*(x-1)^2 s.t. 0 <= x <= 2.0
    qp_eval_f_cb(x) = 0.5*(x[1]-1)^2
    function qp_eval_grad_f_cb(x, grad_f)
        grad_f[1] = x[1] - 1
    end
    function qp_eval_h_cb(x, mode, rows, cols, obj_factor, lambda, values)
        if mode == :Structure
            rows[1] = 1
            cols[1] = 1
        else
            values[1] = 1.0
        end
    end

    x_l = zeros(1)
    x_u = zeros(1)
    x_u[1] = 2.0
    obj = 0.0

    prob = ExaTron.createProblem(1, x_l, x_u, 1, qp_eval_f_cb, qp_eval_grad_f_cb, qp_eval_h_cb)
    @testset "Tron: Julia" begin
        ExaTron.solveProblem(prob)
        @test prob.f == obj
        @test prob.x[1] == 1.0
    end

    if ExaTron.has_c_library()
        @testset "Tron: Fortran" begin
            ExaTron.setOption(prob, "tron_code", :Fortran)
            ExaTron.solveProblem(prob)
            @test prob.f == obj
            @test prob.x[1] == 1.0
        end
    end
end

