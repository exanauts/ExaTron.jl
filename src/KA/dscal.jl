@inline function ExaTron.dscal(n::Int,da::Float64,dx,incx::Int,tx)

    # Ignore incx for now.
    if tx <= n
        @inbounds dx[tx] = da*dx[tx]
    end
    @synchronize

    return
end
