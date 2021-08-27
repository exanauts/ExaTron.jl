function dtsol(n, L::TronSparseMatrixCSC, r)
    zero = 0.0

    # Solve L'*x = r and store the result in r.

    r[n] = r[n]/L.diag_vals[n]
    @inbounds for j=n-1:-1:1
        temp = zero
        for k=L.colptr[j]:L.colptr[j+1]-1
            temp = temp + L.tril_vals[k]*r[L.rowval[k]]
        end
        r[j] = (r[j] - temp)/L.diag_vals[j]
    end

    return
end

function dtsol(n, L::TronDenseMatrix, r)
    zero = 0.0

    # Solve L'*x = r and store the result in r.

    r[n] = r[n]/L.vals[n,n]
    @inbounds for j=n-1:-1:1
        temp = zero
        for k=j+1:n
            temp = temp + L.vals[k,j]*r[k]
        end
        r[j] = (r[j] - temp)/L.vals[j,j]
    end

    return
end

@inline function dtsol(n::Int, L,
                       r,
                       I, J)
    # Solve L'*x = r and store the result in r.

    tx = J
    ty = 1

    @inbounds for j=n:-1:1
        if tx == 1 && ty == 1
            r[j] = r[j] / L[j,j]
        end
        @synchronize

        if tx < j && ty == 1
            r[tx] = r[tx] - L[tx,j]*r[j]
        end
        @synchronize
    end

    return
end