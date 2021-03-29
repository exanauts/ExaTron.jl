function tron_dscal(n,da,dx,incx)

    # Scales a vector by a constant.
    # It uses unrolled loops for increment equal to one.
    # Jack Dongarra, LINPACK, 3/11/78.
    # Modified 3/93 to return if incx <= 0.
    # Modified 12/3/93, array(1) declarations changed to array(*)

    if n <= 0 || incx <= 0
        return
    end

    if incx != 1

        # Code for increment not equal to 1

        nincx = n*incx
        for i=1:incx:nincx
            dx[i] = da*dx[i]
        end

        return

    else

        # Code for increment equal to 1
        # Clean-up loops

        m = mod(n,5)
        if m != 0
            for i=1:m
                dx[i] = da*dx[i]
            end
            if n < 5
                return
            end
        end

        mp1 = m + 1
        for i=mp1:5:n
            dx[i] = da*dx[i]
            dx[i + 1] = da*dx[i + 1]
            dx[i + 2] = da*dx[i + 2]
            dx[i + 3] = da*dx[i + 3]
            dx[i + 4] = da*dx[i + 4]
        end

        return
    end
end

if isequal(BLAS_LIBRARY, :OpenBlas)
    dscal(n,da,dx,incx) = BLAS.scal!(n, da, dx, incx)
    dscal(n,da,dx::CuArray,incx) = CUBLAS.scal!(n, da, dx)
else
    dscal(n,da,dx,incx) = tron_dscal(n, da, dx, incx)
end

@inline function dscal(n::Int,da::Float64,dx::CuDeviceArray{Float64,1},incx::Int)
    tx = threadIdx().x
    ty = threadIdx().y

    # Ignore incx for now.
    if tx <= n && ty == 1
        @inbounds dx[tx] = da*dx[tx]
    end
    CUDA.sync_threads()

    return
end