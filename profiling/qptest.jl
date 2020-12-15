using ExaTron
using Random
using LinearAlgebra
using SparseArrays
using CUDA
using CUDA.CUSPARSE

using Profile
using PProf

function build_problem(; n=10, gpu=true)
    Random.seed!(1)
    m = 0
    P = sparse(Diagonal(rand(n)) + 2.0 * sparse(I, n, n))
    q = randn(n)
    u =   1. * rand(n)
    l = - 100. * rand(n)
    Iz, Jz, vals = findnz(P)

    if gpu
        P = CuSparseMatrixCSC(P)
        q = CuArray(q)
        u = CuArray(u)
        l = CuArray(l)
    end

    eval_f(x) = 0.5 * x' * P * x + q' * x

    function eval_g(x, g)
        fill!(g, 0)
        mul!(g, P, x)
        g .+= q
    end

    function eval_h(x, mode, rows, cols, obj_factor, lambda, values)
        if mode == :Structure
            @inbounds for i in 1:nnz(P)
                rows[i] = Iz[i]
                cols[i] = Jz[i]
            end
        else
            copy!(values, vals)
        end
    end

    return ExaTron.createProblem(n, l, u, nnz(P), eval_f, eval_g, eval_h)
end

prob = build_problem(; n=1000)
prob.x .= 0.5 .* (prob.x_l .+ prob.x_u)
@profile ExaTron.solveProblem(prob)
pprof()

# CUDA.@profile ExaTron.solveProblem(prob)
