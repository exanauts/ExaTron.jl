"""
Subroutine dicfs

Given a symmetric matrix A in compreessed column storage, this
subroutine computes an incomplete Cholesky factor of A + alpha*D,
where alpha is a shift and D is the diagonal matrix with entries
set to the l2 norms of the columns of A.

MINPACK-2 Project. October 1998.
Argonne National Laboratory.
Chih-Jen Lin and Jorge J. More'.
"""
function dicfs(n, nnz, A, L,
               p, alpha, iwa, wa1, wa2)
    nbmax = 3
    alpham = 1.0e-3
    nbfactor = 512

    zero = 0.0
    one = 1.0
    two = 2.0

    # Compute the l2 norms of the columns of A.
    nrm2!(wa1, A, n)

    # Compute the scaling matrix D.
    @inbounds for i=1:n
        if wa1[i] > zero
            wa2[i] = one/sqrt(wa1[i])
        else
            wa2[i] = one
        end
    end

    # Determine a lower bound for the step.

    if alpha <= zero
        alphas = alpham
    else
        alphas = alpha
    end

    # Compute the initial shift.

    alpha = zero
    @inbounds for i=1:n
        if getdiagvalue(A,i) == zero
            alpha = alphas
        else
            alpha = max(alpha,-getdiagvalue(A,i)*(wa2[i]^2))
        end
    end

    if alpha > 0
        alpha = max(alpha,alphas)
    end

    # Search for an acceptable shift. During the search we decrease
    # the lower bound alphas until we determine a lower bound that
    # is not acceptaable. We then increase the shift.
    # The lower bound is decreased by nbfactor at most nbmax times.

    nb = 1
    while true

        if isa(A, TronSparseMatrixCSC)
            # Copy the sparsity structure of A into L.
            @inbounds for i=1:n+1
                L.colptr[i] = A.colptr[i]
            end
            @inbounds for i=1:nnz
                L.rowval[i] = A.rowval[i]
            end

            # Scale A and store in the lower triangular matrix L.
            @inbounds for j=1:n
                L.diag_vals[j] = A.diag_vals[j]*(wa2[j]^2) + alpha
            end
            @inbounds for j=1:n
                for i=A.colptr[j]:A.colptr[j+1]-1
                    L.tril_vals[i] = A.tril_vals[i]*wa2[j]*wa2[A.rowval[i]]
                end
            end
        else
            # TronDenseMatrix case.
            @inbounds for j=1:n
                L.vals[j,j] = A.vals[j,j]*(wa2[j]^2) + alpha
            end
            @inbounds for j=1:n,k=j+1:n
                L.vals[k,j] = A.vals[k,j]*wa2[j]*wa2[k]
            end
        end

        # Attempt an incomplete factorization.
        info = dicf(n,nnz,L, p,
                   view(iwa,1:n),view(iwa,n+1:2*n),
                   view(iwa,2*n+1:3*n),wa1)

        # If the factorization exists, then test for termination.
        # Otherwise increment the shift.
        if info >= 0
            # If the shift is at the lower bound, reduce the shift.
            # Otherwise undo the scaling of L and exit.
            if alpha == alphas && nb < nbmax
                alphas /= nbfactor
                alpha = alphas
                nb = nb + 1
            else
                if isa(L, TronSparseMatrixCSC)
                    @inbounds for i=1:n
                        L.diag_vals[i] /= wa2[i]
                    end
                    @inbounds for j=1:L.colptr[n+1]-1
                        L.tril_vals[j] = L.tril_vals[j]/wa2[L.rowval[j]]
                    end
                else
                    @inbounds for j=1:n,k=j:n
                        L.vals[k,j] /= wa2[k]
                    end
                end
                return
            end
        else
            alpha = max(two*alpha,alphas)
        end
    end

    return
end

@inline function dicfs(n::Int, alpha::Float64, A::CuDeviceArray{Float64,2},
                       L::CuDeviceArray{Float64,2},
                       wa1::CuDeviceArray{Float64,1},
                       wa2::CuDeviceArray{Float64,1})
    tx = threadIdx().x
    ty = threadIdx().y

    nbmax = 3
    alpham = 1.0e-3
    nbfactor = 512

    zero = 0.0
    one = 1.0
    two = 2.0

    # Compute the l2 norms of the columns of A.
    nrm2!(wa1, A, n)

    # Compute the scaling matrix D.
    if tx <= n && ty == 1
        @inbounds wa2[tx] = (wa1[tx] > zero) ? one/sqrt(wa1[tx]) : one
    end
    CUDA.sync_threads()

    # Determine a lower bound for the step.

    if alpha <= zero
        alphas = alpham
    else
        alphas = alpha
    end

    # Compute the initial shift.

    smem_alpha = @cuStaticSharedMem(Float64, 1)

    alpha = zero
    if (tx + blockDim().x * (ty - 1)) <= 32
        if tx <= n && ty == 1
            @inbounds alpha = (A[tx,tx] == zero) ? alphas : max(alpha, -A[tx,tx]*(wa2[tx]^2))
        end

        if blockDim().x > 16
            offset = 16
        elseif blockDim().x > 8
            offset = 8
        elseif blockDim().x > 4
            offset = 4
        elseif blockDim().x > 2
            offset = 2
        else
            offset = 1
        end

        while offset > 0
            alpha = max(alpha, CUDA.shfl_down_sync(0xffffffff, alpha, offset))
            offset >>= 1
        end

        if tx == 1 && ty == 1
            if alpha > 0
                alpha = max(alpha,alphas)
            end
            smem_alpha[1] = alpha
        end
    end

    CUDA.sync_threads()
    alpha = smem_alpha[1]
    CUDA.sync_threads()

    # Search for an acceptable shift. During the search we decrease
    # the lower bound alphas until we determine a lower bound that
    # is not acceptaable. We then increase the shift.
    # The lower bound is decreased by nbfactor at most nbmax times.

    nb = 1
    info = 0

    while true
        if tx <= n && ty == 1
            @inbounds for j=1:n
                L[j,tx] = A[j,tx] * wa2[j] * wa2[tx]
            end
            if alpha != zero
                @inbounds L[tx,tx] += alpha
            end
        end
        CUDA.sync_threads()

        # Attempt a Cholesky factorization.
        info = dicf(n, L)

        # If the factorization exists, then test for termination.
        # Otherwise increment the shift.
        if info >= 0
            # If the shift is at the lower bound, reduce the shift.
            # Otherwise undo the scaling of L and exit.
            if alpha == alphas && nb < nbmax
                alphas /= nbfactor
                alpha = alphas
                nb = nb + 1
            else
                if tx <= n && ty == 1
                    @inbounds for j=1:n
                        if tx >= j
                            L[tx,j] /= wa2[tx]
                            L[j,tx] = L[tx,j]
                        end
                    end
                end
                CUDA.sync_threads()
                return
            end
        else
            alpha = max(two*alpha,alphas)
        end
    end

    return
end