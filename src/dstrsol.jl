"""
Subroutine dstrsol

This subroutine solves the triangular systems L*x = r or L'*x = r.

MINPACK-2 Project. May 1998.
Argonne National Laboratory.
"""
function dstrsol(n,l,ldiag,jptr,indr,r,task)
    zero = 0.0

    # Solve L*x = r and store the result in r.

    if task == 'N'
        for j=1:n
            temp = r[j]/ldiag[j]
            for k=jptr[j]:jptr[j+1]-1
                r[indr[k]] = r[indr[k]] - l[k]*temp
            end
            r[j] = temp
        end

        return
    end

    # Solve L'*x = r and store the result in r.

    if task == 'T'
        r[n] = r[n]/ldiag[n]
        for j=n-1:-1:1
            temp = zero
            for k=jptr[j]:jptr[j+1]-1
                temp = temp + l[k]*r[indr[k]]
            end
            r[j] = (r[j] - temp)/ldiag[j]
        end

        return
    end
end