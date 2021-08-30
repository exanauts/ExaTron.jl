using ExaTron
using Test
using Random
using LinearAlgebra
using SparseArrays
using StatsBase
using CUDA
using KernelAbstractions
using CUDAKernels
const KA = KernelAbstractions

@testset "Test ExaTron" begin
    if has_cuda_gpu()
        @testset "gputest" begin
            include("gputest.jl")
        end
    end
    @testset "qptests" begin
        include("qptest.jl")
    end
    @testset "densetest" begin
        include("densetest.jl")
    end
    @testset "proxal_wrapper" begin
        include("proxal_wrapper.jl")
    end
    @testset "admmtest" begin
        include("admmtest.jl")
    end
end

# include("qptest.jl")
# include("densetest.jl")
# include("admmtest.jl")
