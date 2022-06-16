using KernelAbstractions
using AMDGPU
using ROCKernels
using CUDA
using CUDAKernels
using LinearAlgebra
using Printf
using PowerModels
using ExaTron
using Test
const KA = KernelAbstractions
include("../examples/admm/opfdata.jl")
include("../examples/admm/environment.jl")
include("../examples/admm/generator_kernel.jl")
include("../examples/admm/eval_kernel.jl")
include("../examples/admm/polar_kernel.jl")
include("../examples/admm/bus_kernel.jl")
include("../examples/admm/tron_kernel.jl")
include("../examples/admm/acopf_admm_gpu.jl")

CASE = joinpath("..", "examples", "case9.m")

function one_level_admm(case::String, device::KA.Device)
    # NB: Need to run almost 2,000 iterations to reach convergence with this
    # set of parameters.
    env = admm_gpu(
        case;
        verbose=1,
        iterlim=2000,
        rho_pq=400.0,
        rho_va=40000.0,
        scale=1e-4,
        device=device
    )
    @test isa(env, AdmmEnv)

    model = env.model
    ngen = model.gen_mod.ngen

    par = env.params
    sol = env.solution

    # Check results
    pg = active_power_generation(env)
    qg = reactive_power_generation(env)

    @test sol.status == HAS_CONVERGED
    # Test with solution returned by PowerModels + Ipopt
    @test sol.objval ≈ 5296.6862 rtol=1e-4
    @test (pg |> Array) ≈ [0.897987, 1.34321, 0.941874] rtol=1e-4
    @test (qg |> Array) ≈ [0.1296564, 0.00031842, -0.226342] rtol=1e-2

    # Test restart API
    admm_restart!(env)
    return nothing
end
