function dnsol(n, L::TronSparseMatrixCSC, r)
    # Solve L*x = r and store the result in r.

    @inbounds for j=1:n
        temp = r[j]/L.diag_vals[j]
        for k=L.colptr[j]:L.colptr[j+1]-1
            r[L.rowval[k]] = r[L.rowval[k]] - L.tril_vals[k]*temp
        end
        r[j] = temp
    end

    return
end

function dnsol(n, L::TronDenseMatrix, r)
    # Solve L*x = r and store the result in r.

    @inbounds for j=1:n
        temp = r[j]/L.vals[j,j]
        for k=j+1:n
            r[k] = r[k] - L.vals[k,j]*temp
        end
        r[j] = temp
    end
end

@inline function dnsol(n::Int, L,
                       r,
                       I, J)
    # Solve L*x = r and store the result in r.

    tx = J
    ty = 1

    @inbounds for j=1:n
        if tx == 1 && ty == 1
            r[j] = r[j] / L[j,j]
        end
        @synchronize

        if tx > j && tx <= n && ty == 1
            r[tx] = r[tx] - L[tx,j]*r[j]
        end
        @synchronize
    end

    return
end