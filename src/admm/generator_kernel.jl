# GPU implementation of generator update of ADMM.
function generator_kernel(
    baseMVA, ngen, gen_start,
    u, v, l, rho, pgmin, pgmax, qgmin, qgmax, c2, c1
)
    I = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if (I <= ngen)
        pg_idx = gen_start + 2*(I-1)
        qg_idx = gen_start + 2*(I-1) + 1
        u[pg_idx] = max(pgmin[I],
                        min(pgmax[I],
                            (-(c1[I]*baseMVA + l[pg_idx] -
                            rho[pg_idx]*v[pg_idx])) / (2*c2[I]*(baseMVA^2) + rho[pg_idx])))
        u[qg_idx] = max(qgmin[I],
                        min(qgmax[I],
                            (-(l[qg_idx] - rho[qg_idx]*v[qg_idx])) / rho[qg_idx]))
    end

    return
end

function generator_kernel_cpu(
    baseMVA, ngen, gen_start,
    u, v, l, rho, pgmin, pgmax, qgmin, qgmax, c2, c1
)
    for I=1:ngen
        pg_idx = gen_start + 2*(I-1)
        qg_idx = gen_start + 2*(I-1) + 1
        u[pg_idx] = max(pgmin[I],
                        min(pgmax[I],
                            (-(c1[I]*baseMVA + l[pg_idx] -
                               rho[pg_idx]*v[pg_idx])) / (2*c2[I]*(baseMVA^2) + rho[pg_idx])))
        u[qg_idx] = max(qgmin[I],
                        min(qgmax[I],
                            (-(l[qg_idx] - rho[qg_idx]*v[qg_idx])) / rho[qg_idx]))
    end

    return
end

function generator_kernel_two_level(
    baseMVA, ngen, gen_start,
    u, x, z, l, rho, pgmin, pgmax, qgmin, qgmax, c2, c1
)
    I = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if (I <= ngen)
        pg_idx = gen_start + 2*(I-1)
        qg_idx = gen_start + 2*(I-1) + 1
        u[pg_idx] = max(pgmin[I],
                        min(pgmax[I],
                            (-(c1[I]*baseMVA + l[pg_idx] + rho[pg_idx]*(-x[pg_idx] + z[pg_idx]))) / (2*c2[I]*(baseMVA^2) + rho[pg_idx])))
        u[qg_idx] = max(qgmin[I],
                        min(qgmax[I],
                            (-(l[qg_idx] + rho[qg_idx]*(-x[qg_idx] + z[qg_idx]))) / rho[qg_idx]))
    end

    return
end

function generator_kernel_two_level_cpu(
    baseMVA, ngen, gen_start,
    u, x, z, l, rho, pgmin, pgmax, qgmin, qgmax, c2, c1
)
    for I=1:ngen
        pg_idx = gen_start + 2*(I-1)
        qg_idx = gen_start + 2*(I-1) + 1
        u[pg_idx] = max(pgmin[I],
                        min(pgmax[I],
                            (-(c1[I]*baseMVA + l[pg_idx] + rho[pg_idx]*(-x[pg_idx] + z[pg_idx]))) / (2*c2[I]*(baseMVA^2) + rho[pg_idx])))
        u[qg_idx] = max(qgmin[I],
                        min(qgmax[I],
                            (-(l[qg_idx] + rho[qg_idx]*(-x[qg_idx] + z[qg_idx]))) / rho[qg_idx]))
    end

    return
end

function generator_kernel_proxal_first_period(
    baseMVA, ngen, gen_start,
    u, x, z, l, rho, pgmin, pgmax, qgmin, qgmax, c2, c1,
    tau_prox, rho_prox, p_ref_prox, l_next_prox, pg_next_prox
)
    I = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if I <= ngen
        pg_idx = gen_start + 2*(I-1)
        qg_idx = gen_start + 2*(I-1) + 1

        orig_b = c1[I]*baseMVA + l[pg_idx] + rho[pg_idx]*(-x[pg_idx] + z[pg_idx])
        prox_b = -tau_prox*p_ref_prox[I] + l_next_prox[I] + rho_prox*(-pg_next_prox[I])
        u[pg_idx] = max(pgmin[I],
                        min(pgmax[I],
                            (-(orig_b + prox_b) / (2*c2[I]*(baseMVA^2) + rho[pg_idx] + tau_prox + rho_prox))))
        u[qg_idx] = max(qgmin[I],
                        min(qgmax[I],
                            (-(l[qg_idx] + rho[qg_idx]*(-x[qg_idx] + z[qg_idx]))) / rho[qg_idx]))
    end
end

function generator_kernel_proxal_in_between_period(
    baseMVA, ngen, gen_start,
    u, x, z, l, rho, pgmin, pgmax, qgmin, qgmax, c2, c1,
    tau_prox, rho_prox, p_ref_prox, l_next_prox, pg_next_prox,
    l_prev_prox, pg_prev_prox, smax, smin
)
    I = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if I <= ngen
        pg_idx = gen_start + 2*(I-1)
        qg_idx = gen_start + 2*(I-1) + 1

        # Fill out the solution of qg first.
        u[qg_idx] = max(qgmin[I],
                        min(qgmax[I],
                            (-(l[qg_idx] + rho[qg_idx]*(-x[qg_idx] + z[qg_idx]))) / rho[qg_idx]))

        orig_b = c1[I]*baseMVA + l[pg_idx] + rho[pg_idx]*(-x[pg_idx] + z[pg_idx])
        prox_b = -tau_prox*p_ref_prox[I] + l_next_prox[I] + rho_prox*(-pg_next_prox[I])
        pg_val1 = max(pgmin[I],
                     min(pgmax[I],
                         -(orig_b + prox_b) / (2*c2[I]*(baseMVA)^2 + rho[pg_idx] + tau_prox + rho_prox)))
        s_val1 = -(l_prev_prox[I] + rho_prox*(pg_prev_prox[I] - pg_val1)) / rho_prox

        s_val2 = smax[I]
        prox_b = -tau_prox*p_ref_prox[I] - (l_prev_prox[I] + rho_prox*(pg_prev_prox[I] + s_val2)) + l_next_prox[I] + rho_prox*(-pg_next_prox[I])
        pg_val2 = max(pgmin[I],
                      min(pgmax[I],
                          (-(orig_b + prob_b) / (2*c2[I]*(baseMVA)^2 + rho[pg_idx] + tau_prox + 2*rho_prox))))
        lg_val2 = l_prev_prox[I] + rho_prox*(pg_prev_prox[I] - pg_val2 + s_val2)

        s_val3 = smin[I]
        prox_b = -tau_prox*p_ref_prox[I] - (l_prev_prox[I] + rho_prox*(pg_prev_prox[I] + s_val3)) + l_next_prox[I] + rho_prox*(-pg_next_prox[I])
        pg_val3 = max(pgmin[I],
                      min(pgmax[I],
                          (-(orig_b + prob_b) / (2*c2[I]*(baseMVA)^2 + rho[pg_idx] + tau_prox + 2*rho_prox))))
        lg_val3 = l_prev_prox[I] + rho_prox*(pg_prev_prox[I] - pg_val3 + s_val3)

        if smin[I] <= s_val1 && s_val1 >= smax[I]
            u[pg_idx] = pg_val1
            # Need to return a solution for s.
        elseif lg_val2 <= 0
            u[pg_idx] = pg_val2
            # Need to return a solution for s.
        elseif lg_val3 >= 0
            u[pg_idx] = pg_val3
            # Need to return a solution for s.
        else
            # ERROR!
        end
    end
end

function generator_kernel_proxal_last_period(
    baseMVA, ngen, gen_start,
    u, x, z, l, rho, pgmin, pgmax, qgmin, qgmax, c2, c1,
    tau_prox, rho_prox, p_ref_prox, l_prev_prox, pg_prev_prox, smax, smin
)
    I = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if I <= ngen
        pg_idx = gen_start + 2*(I-1)
        qg_idx = gen_start + 2*(I-1) + 1

        # Fill out the solution of qg first.
        u[qg_idx] = max(qgmin[I],
                        min(qgmax[I],
                            (-(l[qg_idx] + rho[qg_idx]*(-x[qg_idx] + z[qg_idx]))) / rho[qg_idx]))

        orig_b = c1[I]*baseMVA + l[pg_idx] + rho[pg_idx]*(-x[pg_idx] + z[pg_idx])
        prox_b = -tau_prox*p_ref_prox[I]
        pg_val1 = max(pgmin[I],
                     min(pgmax[I],
                         -(orig_b + prox_b) / (2*c2[I]*(baseMVA)^2 + rho[pg_idx] + tau_prox)))
        s_val1 = -(l_prev_prox[I] + rho_prox*(pg_prev_prox[I] - pg_val1)) / rho_prox

        s_val2 = smax[I]
        prox_b = -tau_prox*p_ref_prox[I] - (l_prev_prox[I] + rho_prox*(pg_prev_prox[I] + s_val2))
        pg_val2 = max(pgmin[I],
                      min(pgmax[I],
                          (-(orig_b + prob_b) / (2*c2[I]*(baseMVA)^2 + rho[pg_idx] + tau_prox + rho_prox))))
        lg_val2 = l_prev_prox[I] + rho_prox*(pg_prev_prox[I] - pg_val2 + s_val2)

        s_val3 = smin[I]
        prox_b = -tau_prox*p_ref_prox[I] - (l_prev_prox[I] + rho_prox*(pg_prev_prox[I] + s_val3))
        pg_val3 = max(pgmin[I],
                      min(pgmax[I],
                          (-(orig_b + prob_b) / (2*c2[I]*(baseMVA)^2 + rho[pg_idx] + tau_prox + rho_prox))))
        lg_val3 = l_prev_prox[I] + rho_prox*(pg_prev_prox[I] - pg_val3 + s_val3)

        if smin[I] <= s_val1 && s_val1 >= smax[I]
            u[pg_idx] = pg_val1
            # Need to return a solution for s.
        elseif lg_val2 <= 0
            u[pg_idx] = pg_val2
            # Need to return a solution for s.
        elseif lg_val3 >= 0
            u[pg_idx] = pg_val3
            # Need to return a solution for s.
        else
            # ERROR!
        end
    end
end