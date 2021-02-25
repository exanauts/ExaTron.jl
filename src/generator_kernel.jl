# GPU implementation of generator update of ADMM.
function generator_kernel(
    baseMVA, ngen, pg_start, qg_start,
    u, v, l, rho, pgmin, pgmax, qgmin, qgmax, c2, c1, c0
)
    I = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if (I <= ngen)
        u[pg_start+I] = max(pgmin[I],
                              min(pgmax[I],
                            (-(c1[I]*baseMVA + l[pg_start+I] -
                               rho[pg_start+I]*v[pg_start+I])) / (2*c2[I]*(baseMVA^2) + rho[pg_start+I])))
        u[qg_start+I] = max(qgmin[I],
                              min(qgmax[I],
                            (-(l[qg_start+I] - rho[qg_start+I]*v[qg_start+I])) / rho[qg_start+I]))
    end

    return
end

function generator_kernel_cpu(
    baseMVA, ngen, pg_start, qg_start,
    u, v, l, rho, pgmin, pgmax, qgmin, qgmax, c2, c1, c0
)
    for I=1:ngen
        u[pg_start+I] = max(pgmin[I],
                              min(pgmax[I],
                            (-(c1[I]*baseMVA + l[pg_start+I] -
                               rho[pg_start+I]*v[pg_start+I])) / (2*c2[I]*(baseMVA^2) + rho[pg_start+I])))
        u[qg_start+I] = max(qgmin[I],
                              min(qgmax[I],
                            (-(l[qg_start+I] - rho[qg_start+I]*v[qg_start+I])) / rho[qg_start+I]))
    end

    return
end