@inline function ExaTron.dnsol(n::Int, L,
                       r,
                       tx)
    # Solve L*x = r and store the result in r.
    @inbounds for j=1:n
        if tx == 1
            r[j] = r[j] / L[j,j]
        end
        @synchronize

        if tx > j && tx <= n
            r[tx] = r[tx] - L[tx,j]*r[j]
        end
        @synchronize
    end

    return
end
