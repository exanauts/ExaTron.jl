"""
Subroutine dstrsol

This subroutine solves the triangular systems L*x = r or L'*x = r.

MINPACK-2 Project. May 1998.
Argonne National Laboratory.
"""
function dstrsol(n, L, r, task)
    zero = zero(eltype(r))

    # Solve L*x = r and store the result in r.

    if task == 'N'
        @inbounds for j=1:n
            temp = r[j]/L.diag_vals[j]
            for k=L.colptr[j]:L.colptr[j+1]-1
                r[L.rowval[k]] = r[L.rowval[k]] - L.tril_vals[k]*temp
            end
            r[j] = temp
        end

        return
    end

    # Solve L'*x = r and store the result in r.

    if task == 'T'
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
end
