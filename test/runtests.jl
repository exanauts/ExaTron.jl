using ExaTron
using Test
using Random
using LinearAlgebra
using SparseArrays
using StatsBase
using CUDA

try
    include("gputest.jl")
catch e
    println(e)
end

include("qptest.jl")
include("densetest.jl")
