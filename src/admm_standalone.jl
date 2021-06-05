using ExaTron

datafile = ARGS[1]
pq_val = parse(Float64, ARGS[2])
va_val = parse(Float64, ARGS[3])
max_iter = parse(Int, ARGS[4])
gpu = parse(Bool, ARGS[5])

println("Data: ", datafile, " rho_pq = ", pq_val, " rho_va = ", va_val, " max_iter = ", max_iter, " gpu = ", gpu)

# Warm-up
ExaTron.admm_rect_gpu(datafile;
                      iterlim=1, rho_pq=pq_val, rho_va=va_val, scale=1e-4, use_polar=true, use_gpu=gpu)

# Run
ExaTron.admm_rect_gpu(datafile;
                      iterlim=max_iter, rho_pq=pq_val, rho_va=va_val, scale=1e-4, use_polar=true, use_gpu=gpu)
