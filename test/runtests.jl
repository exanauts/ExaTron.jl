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
        devices = Vector()
        push!(devices, KA.CPU())
        if CUDA.has_cuda_gpu() || AMDGPU.has_rocm_gpu()
            include("KA.jl")
            if CUDA.has_cuda_gpu()
                push!(devices, CUDABackend())
            end
            if AMDGPU.has_rocm_gpu()
                push!(devices, ROCBackend())
            end
        end
        @testset "Testing one-level ADMM using $device" for device in devices
            include("admmtest.jl")
            one_level_admm(CASE, device)
        end
    end
end
