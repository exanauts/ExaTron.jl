@inline function ExaTron.dcopy(n::Int,dx,incx::Int,
                       dy,incy::Int,
                       tx)
    # Ignore incx and incy for now.
    if tx <= n
        @inbounds dy[tx] = dx[tx]
    end
    @synchronize

    return
end
