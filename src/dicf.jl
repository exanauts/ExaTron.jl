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
    indices k with i = L.rowval[indf[k]].

For the computation of the j-th column, the array indr records
the row indices. Hence, if nlj is the number of nonzeros in the
j-th column, then indr[1],...,indr[nlj] are the row indices.
Also, for i > j, indf[i] marks the row indices in the j-th
column so that indf[i] = 1 if l[i,j] is not zero.

MINPACK-2 Project. May 1998.
Argonne National Laboratory.
Chih-Jen Lin and Jorge J. More'.
"""
function dicf(n, nnz, L::TronSparseMatrixCSC, p, indr, indf, list, w)
    zero = 0.0
    insortf = 20

    info = 0
    @inbounds for j=1:n
        indf[j] = 0
        list[j] = 0
    end

    # Make room for L by moving A to the last n*p positions in a.

    np = n*p
    @inbounds for j=1:n+1
        L.colptr[j] = L.colptr[j] + np
    end
    @inbounds for j=nnz:-1:1
        L.rowval[np+j] = L.rowval[j]
        L.tril_vals[np+j] = L.tril_vals[j]
    end

    # Compute the incomplete Cholesky factorization.

    isj = L.colptr[1]
    L.colptr[1] = 1
    @inbounds for j=1:n

        # Load column j into the array w. The first and last elements
        # of the j-th column of A are a(isj) and a(iej).

        nlj = 0
        iej = L.colptr[j+1] - 1

        @inbounds for ip in isj:iej
            i = L.rowval[ip]
            w[i] = L.tril_vals[ip]
            nlj = nlj + 1
            indr[nlj] = i
            indf[i] = 1
        end

        # Exit if the current pivot is not positive.
        if L.diag_vals[j] <= zero
            info = -j
            return info
        end
        L.diag_vals[j] = sqrt(L.diag_vals[j])

        # Update column j using the previous columns.
        k = list[j]
        while k != 0
            isk = indf[k]
            iek = L.colptr[k+1]-1

            # Set lval to l[j,k].
            lval = L.tril_vals[isk]

            # Update indf and list.

            newk = list[k]
            isk = isk + 1
            if isk < iek
                indf[k] = isk
                list[k] = list[L.rowval[isk]]
                list[L.rowval[isk]] = k
            end
            k = newk

            # Compute the update a[i,i] <- a[i,j] - l[i,k]*l[j,k].
            # In this loop we pick up l[i,k] for k < j and i > j.
            @inbounds for ip=isk:iek
                i = L.rowval[ip]
                if indf[i] != 0
                    w[i] = w[i] - lval*L.tril_vals[ip]
                else
                    indf[i] = 1
                    nlj = nlj + 1
                    indr[nlj] = i
                    w[i] = -lval*L.tril_vals[ip]
                end
            end
        end

        # Compute the j-th column of L.
        @inbounds for k=1:nlj
            w[indr[k]] = w[indr[k]]/L.diag_vals[j]
        end

        # Set mlj to the number of nonzeros to be retained.
        mlj = min(iej-isj+1+p,nlj)
        kth = nlj - mlj + 1

        if nlj >= 1
            # Determine the kth smallest elements in the current column
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
        # of the j-th column of L are L.tril_vals[newisj] and L.tril_vals[newiej].
        newisj = L.colptr[j]
        newiej = newisj + mlj - 1
        @inbounds for k in newisj:newiej
            L.tril_vals[k] = w[indr[k-newisj+kth]]
            L.rowval[k] = indr[k-newisj+kth]
        end

        # Update the diagonal elements.
        @inbounds for k=kth:nlj
            L.diag_vals[indr[k]] = L.diag_vals[indr[k]] - w[indr[k]]^2
        end

        # Update indr and list for the j-th column.
        if newisj < newiej
            indf[j] = newisj
            list[j] = list[L.rowval[newisj]]
            list[L.rowval[newisj]] = j
        end

        # Clear out elements j+1,...,n of the array indf.
        @inbounds for k=1:nlj
            indf[indr[k]] = 0
        end

        # Update isj and L.colptr.
        isj = L.colptr[j+1]
        L.colptr[j+1] = newiej + 1
    end

    return info
end

function dicf(n, nnz, L::TronDenseMatrix, p, indr, indf, list, w)
    zero = 0.0
    info = 0

    # We perform left-looking Cholesky factorization.
    @inbounds for j=1:n
        if L.vals[j,j] <= zero
            info = -j
            return info
        end
        L.vals[j,j] = sqrt(L.vals[j,j])

        # Update column j using the previous columns.
        @inbounds for k=1:j-1,i=j+1:n
            L.vals[i,j] = L.vals[i,j] - L.vals[i,k]*L.vals[j,k]
        end

        # Compute the j-th column of L.
        @inbounds for i=j+1:n
            L.vals[i,j] = L.vals[i,j] / L.vals[j,j]
            L.vals[i,i] = L.vals[i,i] - L.vals[i,j]^2
        end
    end

    return info
end

#=
# Right-looking Cholesky
function dicf(n::Int,L::CuDeviceArray{Float64})
    tx = threadIdx().x
    ty = threadIdx().y

    zero = 0.0
    Lmn = L[tx,ty]
    @inbounds for j=1:n
        # Update the diagonal.
        if (tx == j) && (ty == j)
            if Lmn > zero
                L[j,j] = sqrt(Lmn)
            else
                L[j,j] = -1.0
            end
        end
        CUDA.sync_threads()

        Ljj = L[j,j]
        if Ljj <= zero
            CUDA.sync_threads()
            return -1
        end

        # Update the jth column.
        if ty == j
            L[tx,j] = Lmn / Ljj
        end
        CUDA.sync_threads()

        # Update the trailing submatrix. To avoid if-conditional,
        # we update the whole matrix, but only the trailing part
        # will be saved in next iteration.
        Lmn = Lmn - L[tx,j] * L[ty,j]
    end

    if tx > ty
        L[ty,tx] = L[tx,ty]
    end
    CUDA.sync_threads()

    return 0
end
=#

# Left-looking Cholesky
function dicf(n::Int,L::CuDeviceArray{Float64})
    tx = threadIdx().x
    ty = threadIdx().y

    @inbounds for j=1:n
        # Apply the pending updates.
        if j > 1
            if tx >= j && tx <= n && ty == 1
                @inbounds for k=1:j-1
                    L[tx,j] -= L[tx,k] * L[j,k]
                end
            end
        end
        CUDA.sync_threads()

        if (L[j,j] <= 0)
            CUDA.sync_threads()
            return -1
        end

        Ljj = sqrt(L[j,j])
        if tx >= j && tx <= n && ty == 1
            L[tx,j] /= Ljj
        end
        CUDA.sync_threads()
    end

    if tx <= n && ty == 1
        @inbounds for j=1:n
            if tx > j
                L[j,tx] = L[tx,j]
            end
        end
    end
    #=
    if tx > ty
        L[ty,tx] = L[tx,ty]
    end
    =#
    CUDA.sync_threads()

    return 0
end