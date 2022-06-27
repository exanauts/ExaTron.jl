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
