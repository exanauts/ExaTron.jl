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
    dnrm2(n, x::CuArray, incx) = CUBLAS.nrm2(n, x)
else
    dnrm2(n,x,incx) = tron_dnrm2(n, x, incx)
end

@inline function dnrm2(n::Int,x::CuDeviceArray{Float64,1},incx::Int)
    tx = threadIdx().x
    ty = threadIdx().y
    smem_v = @cuStaticSharedMem(Float64, 1)

    v = 0.0
    if (tx + blockDim().x * (ty - 1)) <= 32
        if tx <= n && ty == 1
            @inbounds v = x[tx]*x[tx]
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
            v += CUDA.shfl_down_sync(0xffffffff, v, offset)
            offset >>= 1
        end

        if tx == 1 && ty == 1
            v = sqrt(v)
            smem_v[1] = v
        end
    end

    CUDA.sync_threads()
    v = smem_v[1]
    CUDA.sync_threads()

    return v
end