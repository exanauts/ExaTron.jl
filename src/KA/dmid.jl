@inline function ExaTron.dmid(n::Int, x,
                      xl, xu,
                      tx)

    if tx <= n
        @inbounds x[tx] = max(xl[tx], min(x[tx], xu[tx]))
    end
    @synchronize

    return
end
