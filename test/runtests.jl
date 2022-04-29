using ExaTron
using Test
using Random
using LinearAlgebra
using SparseArrays
using StatsBase
using CUDA

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
end
