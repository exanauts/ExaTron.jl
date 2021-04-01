@inline function dnsol(n, L, r)
    # Solve L*x = r and store the result in r.

    tx = threadIdx().x
    ty = threadIdx().y

    @inbounds for j=1:n
        if tx == 1 && ty == 1
            r[j] = r[j] / L[j,j]
        end
        CUDA.sync_threads()

        if tx > j && tx <= n && ty == 1
            r[tx] = r[tx] - L[tx,j]*r[j]
        end
        CUDA.sync_threads()
    end

    return
end