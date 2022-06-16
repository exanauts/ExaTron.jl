@inline function ExaTron.dnrm2(n::Int,x,incx::Int, tx)

    @synchronize
    v = 0.0
    for i in 1:n
        @inbounds v += x[i]*x[i]
    end

    @synchronize
    v = sqrt(v)
    @synchronize

    return v
end
