function dgpnorm(n::Int, x::Array{Float64}, xl::Array{Float64},
                 xu::Array{Float64}, g::Array{Float64})
    inf_norm = 0.0
    for i=1:n
        if xl[i] != xu[i]
            if x[i] == xl[i]
                v = (min(g[i], 0.0))^2
            elseif x[i] == xu[i]
                v = (max(g[i], 0.0))^2
            else
                v = g[i]^2
            end

            v = sqrt(v)
            inf_norm = (inf_norm > v) ? inf_norm : v
        end
    end

    return inf_norm
end

@inline function dgpnorm(n::Int, x, xl,
                         xu, g,
                         I, J)

    tx = J
    ty = 1
    @synchronize
    res = 0.0
    inf_norm = @localmem Float64 (1,)

    v = 0.0
    if tx == 1
        inf_norm[1] = 0.0
        for i in 1:n
            @inbounds begin
                if xl[i] != xu[i]
                    if x[i] == xl[i]
                        v = min(g[i], 0.0)
                        v = v*v
                    elseif x[i] == xu[i]
                        v = max(g[i], 0.0)
                        v = v*v
                    else
                        v = g[i]*g[i]
                    end

                    v = sqrt(v)
                    if inf_norm[1] > v
                        inf_norm[1] = inf_norm[1]
                    else
                        inf_norm[1] = v
                    end
                end
            end
        end
    end

    @synchronize
    if tx <= n
        res = inf_norm[1]
    end
    @synchronize
    return res
end
