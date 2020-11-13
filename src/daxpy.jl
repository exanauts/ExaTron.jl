"""
Subroutine daxpy

This subroutine computes constant times a vector plus a vector.
It uses unrolled loops for increments equal to one.
Jack Dongarra, LINPACK, 3/11/78.
"""
function tron_daxpy(n,da,dx,incx,dy,incy)
    if n <= 0
        return
    end
    if da == 0.0
        return
    end
    if incx != 1 || incy != 1
        ix = 1
        iy = 1
        if incx < 0
            ix = (-n+1)*incx + 1
        end
        if incy < 0
            iy = (-n+1)*incy + 1
        end
        for i=1:n
            dy[iy] += da*dx[ix]
            ix += incx
            iy += incy
        end
        return
    else

        # Code for both increments equal to 1

        m = mod(n, 4)
        if m != 0
            for i=1:m
                dy[i] += da*dx[i]
            end
        end
        if n < 4
            return
        end

        mp1 = m + 1
        for i=mp1:4:n
            dy[i] += da*dx[i]
            dy[i + 1] += da*dx[i + 1]
            dy[i + 2] += da*dx[i + 2]
            dy[i + 3] += da*dx[i + 3]
        end
        return
    end
end

if isequal(BLAS_LIBRARY, :OpenBlas)
    daxpy(n,da,dx,incx,dy,incy) = BLAS.axpy!(da,dx,dy)
else
    daxpy(n,da,dx,incx,dy,incy) = tron_daxpy(n,da,dx,incx,dy,incy)
end

