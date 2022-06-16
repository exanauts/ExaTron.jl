# Left-looking Cholesky
@inline function ExaTron.dicf(n::Int,L,tx)
    @inbounds for j=1:n
        # Apply the pending updates.
        if tx >= j && tx <= n
            for k=1:j-1
                L[tx,j] -= L[tx,k] * L[j,k]
            end
        end
        @synchronize

        if (L[j,j] <= 0)
            @synchronize
            return -1
        end

        Ljj = sqrt(L[j,j])
        if tx >= j && tx <= n
            L[tx,j] /= Ljj
        end
        @synchronize
    end

    if tx <= n
        @inbounds for j=1:n
            if tx > j
                L[j,tx] = L[tx,j]
            end
        end
    end
    @synchronize

    return 0
end
