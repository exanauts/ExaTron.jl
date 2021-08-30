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

try
    include("gputest.jl")
catch e
    println(e)
end

# include("qptest.jl")
# include("densetest.jl")
# include("admmtest.jl")
