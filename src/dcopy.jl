function tron_dcopy(n,dx,incx,dy,incy)
    # Copies a vector, x, to a vector, y.
    # It uses unrolled loops for increments equal to one.
    # Jack Dongarra, LINPACK, 3/11/78.
    # Modified 12/3/93, array(1) declarations changed to array(*)

    if n <= 0
        return
    end
    if incx != 1 || incy != 1

        # Code for unequal increments or equal increments
        # not equal to 1

        ix = 1
        iy = 1
        if incx < 0
            ix = (-n+1)*incx + 1
        end
        if incy < 0
            iy = (-n+1)*incy + 1
        end
        for i=1:n
            dy[iy] = dx[ix]
            ix = ix + incx
            iy = iy + incy
        end

        return
    else

        # Code for both increments equal to 1
        # Clean-up loops

        m = mod(n,7)
        if m != 0
            for i=1:m
                dy[i] = dx[i]
            end
            if n < 7
                return
            end
        end
        mp1 = m + 1
        for i=mp1:7:n
            dy[i] = dx[i]
            dy[i + 1] = dx[i + 1]
            dy[i + 2] = dx[i + 2]
            dy[i + 3] = dx[i + 3]
            dy[i + 4] = dx[i + 4]
            dy[i + 5] = dx[i + 5]
            dy[i + 6] = dx[i + 6]
        end

        return
    end
end

if isequal(BLAS_LIBRARY, :OpenBlas)
    dcopy(n,dx,incx,dy,incy) = copyto!(dy, 1, dx, 1, n)
else
    dcopy(n,dx,incx,dy,incy) = tron_dcopy(n,dx,incx,dy,incy)
end

function dcopy(n::Int,dx::CuDeviceArray{Float64},incx::Int,
               dy::CuDeviceArray{Float64},incy::Int)
    tx = threadIdx().x
    ty = threadIdx().y

    # Ignore incx and incy for now.
    if ty == 1
        dy[tx] = dx[tx]
    end
    CUDA.sync_threads()

    return
end