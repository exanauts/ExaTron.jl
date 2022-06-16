@inline function ExaTron.dgpnorm(n::Int, x, xl,
                         xu, g,
                         tx)

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
    res = inf_norm[1]
    @synchronize
    return res
end
