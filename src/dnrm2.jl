"""
DNRM2 returns the euclidean norm of a vector via the function
name, so that

   DNRM2 := sqrt( x'*x )

-- This version written on 25-October-1982.
   Modified on 14-October-1993 to inline the call to DLASSQ.
   Sven Hammarling, Nag Ltd.
"""
@inline function dnrm2(n, x, incx)
    tx = threadIdx().x

    v = 0.0
    if tx <= n  # No check on ty so that each warp has v.
        @inbounds v = x[tx]*x[tx]
    end

    # shfl_down_sync() will automatically sync threads in a warp.

    offset = 16
    while offset > 0
        v += CUDA.shfl_down_sync(0xffffffff, v, offset)
        offset >>= 1
    end
    v = sqrt(v)
    v = CUDA.shfl_sync(0xffffffff, v, 1)

    return v
end