@inline function dtsol(n, L, r)
    # Solve L'*x = r and store the result in r.

    tx = threadIdx().x
    ty = threadIdx().y

    @inbounds for j=n:-1:1
        if tx == 1 && ty == 1
            r[j] = r[j] / L[j,j]
        end
        CUDA.sync_threads()

        if tx < j && ty == 1
            r[tx] = r[tx] - L[tx,j]*r[j]
        end
        CUDA.sync_threads()
    end

    return
end