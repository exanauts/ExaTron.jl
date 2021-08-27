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

v = ones(10)
shift = 0
ev = kernel2(CPU())(v, shift, ndrange = 10)
wait(ev)
