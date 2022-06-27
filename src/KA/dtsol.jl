@inline function ExaTron.dtsol(n::Int, L,
                       r,
                       tx)
    # Solve L'*x = r and store the result in r.

    @inbounds for j=n:-1:1
        if tx == 1
            r[j] = r[j] / L[j,j]
        end
        @synchronize

        if tx < j
            r[tx] = r[tx] - L[tx,j]*r[j]
        end
        @synchronize
    end

    return
end
