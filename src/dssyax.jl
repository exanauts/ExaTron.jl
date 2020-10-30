"""
Subroutine dssyax

This subroutine computes the matrix-vector product y = A*x,
where A is a symmetric matrix with the strict lower triangular
part in compressed column storage.
"""
function dssyax(n,a,adiag,jptr,indr,x,y)
    zero = 0.0

    for i=1:n
        y[i] = adiag[i]*x[i]
    end

    for j=1:n
        rowsum = zero
        for i=jptr[j]:jptr[j+1]-1
            rowsum += a[i]*x[indr[i]]
            y[indr[i]] += a[i]*x[j]
        end
        y[j] += rowsum
    end

    return
end