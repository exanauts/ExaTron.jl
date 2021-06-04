# Getting Started

We provide a brief overview of the package.

## Installation

This package can be installed by cloning this repository:
```julia
] add https://github.com/exanauts/ExaTron.jl
```

## Example

The following code snippet shows how to use this pacakge to solve a simple quadratic programming problem of the form

```math
\min \; 0.5*(x-1)^2 \; \text{s.t.} \; 0 \leq x \leq 2.0
```

```julia
using ExaTron

# callback function to evaluate objective
qp_eval_f_cb(x) = 0.5*(x[1]-1)^2

# callback function to evaluate the gradient 
function qp_eval_grad_f_cb(x, grad_f)
    grad_f[1] = x[1] - 1
end

# callback function to evaluate the Hessian
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

ExaTron.solveProblem(prob)
```
