using KernelAbstractions
using CUDA
using CUDAKernels
const KA = KernelAbstractions
# using LinearAlgebra

function nested(v, I)
    @synchronize
end

@kernel function kernel2(v, shift::Int)
    I      = @index(Group, Linear)
    J      = @index(Local, Linear)
    @synchronize
    nested(v, I)
end

v = ones(10) |> CuArray
shift = 0
ev = kernel2(CUDADevice())(v, shift, ndrange = 10)
wait(ev)
