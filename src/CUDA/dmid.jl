@inline function ExaTron.dmid(n::Int, x::CuDeviceArray{Float64,1},
                      xl::CuDeviceArray{Float64,1}, xu::CuDeviceArray{Float64,1})
    tx = threadIdx().x

    if tx <= n
        @inbounds x[tx] = max(xl[tx], min(x[tx], xu[tx]))
    end
    CUDA.sync_threads()

    return
end
