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

function dtsol(n::Int, L::CuDeviceArray{Float64},
               r::CuDeviceArray{Float64})
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