using AMDGPU
using KernelAbstractions
using ROCKernels

@kernel function kernel(v)
    I      = @index(Group, Linear)
    J      = @index(Local, Linear)
    v[J] = 2.0
    v[I] = 3.0
end

v = ones(10) |> ROCArray
wait(kernel(ROCDevice())(v, ndrange = 10, dependencies=Event(ROCDevice())))
@show v |> Array
