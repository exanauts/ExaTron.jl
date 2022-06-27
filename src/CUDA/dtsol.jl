@inline function ExaTron.dtsol(n::Int, L::CuDeviceArray{Float64,2},
                       r::CuDeviceArray{Float64,1})
    # Solve L'*x = r and store the result in r.

    tx = threadIdx().x

    @inbounds for j=n:-1:1
        if tx == 1
            r[j] = r[j] / L[j,j]
        end
        CUDA.sync_threads()

        if tx < j
            r[tx] = r[tx] - L[tx,j]*r[j]
        end
        CUDA.sync_threads()
    end

    return
end
