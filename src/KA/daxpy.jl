@inline function ExaTron.daxpy(n::Int,da::Float64,
                       dx,incx::Int,
                       dy,incy::Int,
                       tx)
    if tx <= n
        @inbounds dy[tx] = dy[tx] + da*dx[tx]
    end
    @synchronize

    return
end
