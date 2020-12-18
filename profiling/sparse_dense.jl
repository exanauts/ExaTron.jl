using ExaTron
using LinearAlgebra
using SparseArrays
using Random

using Profile
using PProf

Random.seed!(0)
n = 8
A = ExaTron.TronDenseMatrix{Array}(n)
L = ExaTron.TronDenseMatrix{Array}(n)
for j=1:n, i=j:n
    v = rand(1)[1]
    L.vals[i,j] = v
end
A.vals .= (L.vals * transpose(L.vals))

g = ones(n)
x = zeros(n)
xc = zeros(n)
xl = -Inf*ones(n)
xu = Inf*ones(n)
eval_f_cb(x) = 0.5*(transpose(x)*A.vals*x) + transpose(g)*x
function eval_grad_f_cb(x, grad_f)
    grad_f .= A.vals*x .+ g
end
function eval_h_cb(x, mode, rows, cols, scale, lambda, values)
    if mode == :Structure
        nz = 1
        for j=1:n,i=j:n
            rows[nz] = i
            cols[nz] = j
            nz += 1
        end
    else
        nz = 1
        for j=1:n,i=j:n
            values[nz] = A.vals[i,j]
            nz += 1
        end
    end
end

prob = ExaTron.createProblem(n, xl, xu, Int(floor((n*(n+1))/2)),
                             eval_f_cb, eval_grad_f_cb, eval_h_cb)#; :matrix_type=>:Dense)

howmany = 10
for i=1:howmany
    fill!(prob.x, 0.0)
    @profile ExaTron.solveProblem(prob)
end
pprof()