"""
DNRM2 returns the euclidean norm of a vector via the function
name, so that

   DNRM2 := sqrt( x'*x )

-- This version written on 25-October-1982.
   Modified on 14-October-1993 to inline the call to DLASSQ.
   Sven Hammarling, Nag Ltd.
"""
function tron_dnrm2(n,x,incx)
    one = 1.0
    zero = 0.0

    if n < 1 || incx < 1
        xnorm = zero
    elseif n == 1
        xnorm = abs(x[1])
    else
        scale = zero
        ssq = one

        # The following loop is equivalent to this call to the LAPACK
        # auxiliary routine:
        # CALL DLASSQ( N, X, INCX, SCALE, SSQ )

        for ix=1:incx:1+(n-1)*incx
            if x[ix] != zero
                absxi = abs(x[ix])
                if scale < absxi
                    ssq = one + ssq*(scale/absxi)^2
                    scale = absxi
                else
                    ssq = ssq + (absxi/scale)^2
                end
            end
        end
        xnorm = scale*sqrt(ssq)
    end

    return xnorm
end

if isequal(BLAS_LIBRARY, :OpenBlas)
    dnrm2(n,x,incx) = BLAS.nrm2(n, x, incx)
else
    dnrm2(n,x,incx) = tron_dnrm2(n, x, incx)
end

