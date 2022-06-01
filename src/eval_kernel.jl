# Assume that scaling factor has already been applied to Q and c.
@inline function eval_qp_f_kernel(n::Int, x::CuDeviceArray{Float64,1}, Q::CuDeviceArray{Float64,2}, c::CuDeviceArray{Float64,1})
    # f = xQx/2 + cx
    f = 0.0
    @inbounds begin
        for j=1:n
            for i=1:n
                f += x[i]*Q[i,j]*x[j]
            end
        end
        f *= 0.5
        for j=1:n
            f += c[j]*x[j]
        end
    end
    return f
end

@inline function eval_qp_grad_f_kernel(n::Int, x::CuDeviceArray{Float64,1}, g::CuDeviceArray{Float64,1}, Q::CuDeviceArray{Float64,2}, c::CuDeviceArray{Float64,1})
    # g = Qx + c
    tx = threadIdx().x

    @inbounds begin
        if tx <= n
            g[tx] = c[tx]
        end
        CUDA.sync_threads()
        if tx <= n
            for j=1:n
                g[tx] += Q[tx,j]*x[j]
            end
        end
        CUDA.sync_threads()
    end
    return
end