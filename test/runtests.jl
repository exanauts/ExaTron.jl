using Random
using Test

@testset "ExaTron" begin
    using CUDA
    using AMDGPU
    @testset "CUDA.jl" begin
        if CUDA.has_cuda_gpu()
            include("CUDA.jl")
        end
    end
    @testset "KA.jl" begin
        using KernelAbstractions
        KA = KernelAbstractions
        devices = Vector{KA.Device}()
        push!(devices, KA.CPU())
        if CUDA.has_cuda_gpu() || AMDGPU.has_rocm_gpu()
            include("KA.jl")
            if CUDA.has_cuda_gpu()
                using CUDAKernels
                push!(devices, CUDADevice())
            end
            if AMDGPU.has_rocm_gpu()
                using ROCKernels
                push!(devices, ROCDevice())
            end
        end
        @testset "Testing one-level ADMM using $device" for device in devices
            include("admmtest.jl")
            one_level_admm(CASE, device)
        end
    end
end
