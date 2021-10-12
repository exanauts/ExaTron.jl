
using CUDA
using AMDGPU
using KernelAbstractions
using CUDAKernels
using ROCKernels
using ExaTron
using Test
using LinearAlgebra
using LazyArtifacts
using Test

devices = []
push!(devices, CPU())
if has_cuda_gpu()
    push!(devices, CUDADevice())
end
if AMDGPU.hsa_configured
    push!(devices, ROCDevice())
end

CASE = joinpath(artifact"ExaData", "ExaData", "matpower", "case9.m")

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


@testset "ProxAL wrapper (device=$device)" for device in devices
    data = ExaTron.opf_loaddata(CASE)
    t, T = 1, 2
    rho_pq, rho_va = 400.0, 40000.0
    env = ExaTron.ProxALAdmmEnv(data, device, t, T, rho_pq, rho_va; use_twolevel=true, verbose=0)
    @test isa(env, ExaTron.AdmmEnv)
    @test isa(env.model.gen_mod, ExaTron.ProxALGeneratorModel)

    ExaTron.set_proximal_term!(env, 0.1)
    ExaTron.set_penalty!(env, 0.1)
    ExaTron.set_upper_bound_slack!(env, 2 .* RAMP_AGC)

    ExaTron.set_active_load!(env, LOADS[t]["pd"])
    ExaTron.set_reactive_load!(env, LOADS[t]["qd"])

    pg = ExaTron.active_power_generation(env)

    ExaTron.admm_restart!(env)

    sol = env.solution
    pg = ExaTron.active_power_generation(env)
    pg = pg |> Array
    @test sol.status == ExaTron.HAS_CONVERGED
    @test pg |> Array ≈ [0.8965723471282547, 1.3381394570889165, 0.9386855347032877] rtol=1e-4
end


