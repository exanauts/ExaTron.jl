using KernelAbstractions
using CUDA
using CUDAKernels
const KA = KernelAbstractions
# using LinearAlgebra

function nested2(v, J, p)
    p = 3
    if J == 1
        for i in 1:p
            v[i] = 4.0
        end
    end
end

function nested(v, J, p)
    # @show __ctx__
    p[1] = 2
    if J < 10
        v[J] = p[1]
    end
    @synchronize
    nested2(v, J, p[1])
    @synchronize
end

@kernel function kernel2(v, shift::Int)
    I      = @index(Group, Linear)
    J      = @index(Local, Linear)
    p = @private Float64 (1,)
    @synchronize
    nested(v, J, p)
end

v = ones(10) |> CuArray
shift = 0
ev = kernel2(CUDADevice())(v, shift, ndrange = 10)
wait(ev)
@show v |> Array
