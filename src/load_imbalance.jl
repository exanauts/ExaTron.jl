using DelimitedFiles
using Statistics
using Printf

function load_imbalance(file)
    data = readdlm(file)
    val = data[:,2:end]
    t_max = maximum(val; dims=2)
    t_mean = mean(val; dims=2)
    nu_pk = (t_max ./ t_mean .- 1.0) .* 100.0
    return nu_pk
end

nu_pk = load_imbalance(ARGS[1])
@printf("case = %s\n", ARGS[1])
@printf("nu_max = %.2f nu_min = %.2f nu_mean = %.2f\n", 
         maximum(nu_pk), minimum(nu_pk), mean(nu_pk))
