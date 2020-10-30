"""
Subroutine ddot

This subroutine forms the dot product of two vectors.
It uses unrolled loops for increments equal to one.
Jack Dongarra, LINPACK, 3/11/78.
"""
function ddot(n,dx,incx,dy,incy)
    dtemp = 0.0

    if n <= 0
        return
    end
    if (incx == 1 && incy == 1)
        # Code for both increments equal to 1
        m = mod(n,5)
        if m != 0
            for i=1:m
                dtemp = dtemp + dx[i]*dy[i]
            end
        end
        if n >= 5
            mp1 = m + 1
            for i=mp1:5:n
                dtemp = dtemp + dx[i]*dy[i] + dx[i+1]*dy[i+1] +
                    dx[i+2]*dy[i+2] + dx[i+3]*dy[i+3] + dx[i+4]*dy[i+4]
            end
        end
    else
        ix = 1
        iy = 1
        if incx < 0
            ix = (-n+1)*incx + 1
        end
        if incy < 0
            iy = (-n+1)*incy + 1
        end
        for i=1:n
            dtemp += dx[ix]*dy[iy]
            ix += incx
            iy += incy
        end
    end

    return dtemp
end