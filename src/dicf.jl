"""
Subroutine dicf

Given a sparse symmetric matrix A in compressed row storage,
this subroutine computes an incomplete Cholesky factorization.

Implementation of dicf is based on the Jones-Plassmann code.
Arrays indf and list define the data structure.
At the beginning of the computation of the j-th column,

    For k < j, indf[k] is the index of A for the first
    nonzero l[i,k] in the k-th column with i >= j.

    For k < j, list[i] is a pointer to a linked list of column
    indices k with i = row_ind[indf[k]].

For the computation of the j-th column, the array indr records
the row indices. Hence, if nlj is the number of nonzeros in the
j-th column, then indr[1],...,indr[nlj] are the row indices.
Also, for i > j, indf[i] marks the row indices in the j-th
column so that indf[i] = 1 if l[i,j] is not zero.

MINPACK-2 Project. May 1998.
Argonne National Laboratory.
Chih-Jen Lin and Jorge J. More'.
"""
function dicf(n,nnz,a,diag,col_ptr,row_ind,p,
              indr,indf,list,w)
    zero = 0.0
    insortf = 20

    info = 0
    for j=1:n
        indf[j] = 0
        list[j] = 0
    end

    # Make room for L by moving A to the last n*p positions in a.

    np = n*p
    for j=1:n+1
        col_ptr[j] = col_ptr[j] + np
    end
    for j=nnz:-1:1
        row_ind[np+j] = row_ind[j]
        a[np+j] = a[j]
    end

    # Compute the incomplete Cholesky factorization.

    isj = col_ptr[1]
    col_ptr[1] = 1
    for j=1:n

        # Load column j into the array w. The first and last elements
        # of the j-th column of A are a(isj) and a(iej).

        nlj = 0
        iej = col_ptr[j+1]-1
        for ip=isj:iej
            i = row_ind[ip]
            w[i] = a[ip]
            nlj = nlj + 1
            indr[nlj] = i
            indf[i] = 1
        end

        # Exit if the current pivot is not positive.

        if diag[j] <= zero
            info = -j
            return info
        end
        diag[j] = sqrt(diag[j])

        # Update column j using the previous columns.

        k = list[j]
        while k != 0
            isk = indf[k]
            iek = col_ptr[k+1]-1

            # Set lval to l[j,k].
            lval = a[isk]

            # Update indf and list.

            newk = list[k]
            isk = isk + 1
            if isk < iek
                indf[k] = isk
                list[k] = list[row_ind[isk]]
                list[row_ind[isk]] = k
            end
            k = newk

            # Compute the update a[i,i] <- a[i,j] - l[i,k]*l[j,k].
            # In this loop we pick up l[i,k] for k < j and i > j.

            for ip=isk:iek
                i = row_ind[ip]
                if indf[i] != 0
                    w[i] = w[i] - lval*a[ip]
                else
                    indf[i] = 1
                    nlj = nlj + 1
                    indr[nlj] = i
                    w[i] = -lval*a[ip]
                end
            end
        end

        # Compute the j-th column of L.

        for k=1:nlj
            w[indr[k]] = w[indr[k]]/diag[j]
        end

        # Set mlj to the number of nonzeros to be ratined.

        mlj = min(iej-isj+1+p,nlj)
        kth = nlj - mlj + 1

        if nlj >= 1

            # Determine the kth smallest elements in the current
            # column, and hence, the largest mlj elements.

            dsel2(nlj,w,indr,kth)

            # Sort the row indices of the selected elements. Insertion
            # sort is used for small arrays, and heap sort for larger
            # arrays. The sorting of the row indices is required so that
            # we can retrieve l[i,k] with i > k from indf[k].

            if mlj <= insortf
                insort(mlj,view(indr, kth:n))
            else
                ihsort(mlj,view(indr, kth:n))
            end
        end

        # Store the largest elements in L. The first and last elements
        # of the j-th column of L are a[newisj] and a[newiej].

        newisj = col_ptr[j]
        newiej = newisj + mlj - 1
        for k=newisj:newiej
            a[k] = w[indr[k-newisj+kth]]
            row_ind[k] = indr[k-newisj+kth]
        end

        # Update the diagonal elements.

        for k=kth:nlj
            diag[indr[k]] = diag[indr[k]] - w[indr[k]]^2
        end

        # Update indr and list for the j-th column.

        if newisj < newiej
            indf[j] = newisj
            list[j] = list[row_ind[newisj]]
            list[row_ind[newisj]] = j
        end

        # Clear out elements j+1,...,n of the array indf.

        for k=1:nlj
            indf[indr[k]] = 0
        end

        # Update isj and col_ptr.

        isj = col_ptr[j+1]
        col_ptr[j+1] = newiej + 1
    end

    return info
end