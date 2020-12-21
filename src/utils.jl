
function gpnorm(n, x, x_l, x_u, g)
    two_norm = 0.0
    inf_norm = 0.0

    for i=1:n
        if x_l[i] != x_u[i]
            if x[i] == x_l[i]
                val = (min(g[i], 0.0))^2
            elseif x[i] == x_u[i]
                val = (max(g[i], 0.0))^2
            else
                val = g[i]^2
            end

            two_norm += val
            val = sqrt(val)
            if inf_norm < val
                inf_norm = val
            end
        end
    end

    two_norm = sqrt(two_norm)

    return two_norm, inf_norm
end


