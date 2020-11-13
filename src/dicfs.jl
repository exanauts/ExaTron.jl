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
function dicfs(n,nnz,a,adiag,acol_ptr,arow_ind,
               l,ldiag,lcol_ptr,lrow_ind,
               p,alpha,iwa,wa1,wa2)
    nbmax = 3
    alpham = 1.0e-3
    nbfactor = 512

    zero = 0.0
    one = 1.0
    two = 2.0

    # Compute the l2 norms of the columns of A.

    for i=1:n
        wa1[i] = adiag[i]^2
    end
    for j=1:n
        for i=acol_ptr[j]:acol_ptr[j+1]-1
            k = arow_ind[i]
            wa1[j] = wa1[j] + a[i]^2
            wa1[k] = wa1[k] + a[i]^2
        end
    end
    for j=1:n
        wa1[j] = sqrt(wa1[j])
    end

    # Compute the scaling matrix D.

    for i=1:n
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
    for i=1:n
        if adiag[i] == zero
            alpha = alphas
        else
            alpha = max(alpha,-adiag[i]*(wa2[i]^2))
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

        # Copy the sparsity structure of A into L.

        for i=1:n+1
            lcol_ptr[i] = acol_ptr[i]
        end
        for i=1:nnz
            lrow_ind[i] = arow_ind[i]
        end

        # Scale A and store in the lower triangular matrix L.

        for j=1:n
            ldiag[j] = adiag[j]*(wa2[j]^2) + alpha
        end
        for j=1:n
            for i=acol_ptr[j]:acol_ptr[j+1]-1
                l[i] = a[i]*wa2[j]*wa2[arow_ind[i]]
            end
        end

        # Attempt an incomplete factorization.

        info = dicf(n,nnz,l,ldiag,lcol_ptr,lrow_ind,p,
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
                for i=1:n
                    ldiag[i] /= wa2[i]
                end
                for j=1:lcol_ptr[n+1]-1
                    l[j] = l[j]/wa2[lrow_ind[j]]
                end
                return
            end
        else
            alpha = max(two*alpha,alphas)
        end
    end

    return
end