function generator_kernel_two_level_proxal_first(
    baseMVA::Float64, ngen::Int, gen_start::Int,
    u::CuDeviceArray{Float64,1}, x::CuDeviceArray{Float64,1},
    z::CuDeviceArray{Float64,1}, l::CuDeviceArray{Float64,1},
    rho::CuDeviceArray{Float64,1},
    pgmin::CuDeviceArray{Float64,1}, pgmax::CuDeviceArray{Float64,1},
    qgmin::CuDeviceArray{Float64,1}, qgmax::CuDeviceArray{Float64,1},
    c2::CuDeviceArray{Float64,1}, c1::CuDeviceArray{Float64,1},
    tau_prox::Float64, rho_prox::Float64, pg_ref_prox::CuDeviceArray{Float64,1},
    l_next_prox::CuDeviceArray{Float64,1}, pg_next_prox::CuDeviceArray{Float64,1}
)
    I = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if I <= ngen
        pg_idx = gen_start + 2*(I-1)
        qg_idx = gen_start + 2*(I-1) + 1

        orig_b = c1[I]*baseMVA + l[pg_idx] + rho[pg_idx]*(-x[pg_idx] + z[pg_idx])
        prox_b = -tau_prox*pg_ref_prox[I] + l_next_prox[I] + rho_prox*(-pg_next_prox[I])
        u[pg_idx] = max(pgmin[I],
                        min(pgmax[I],
                            (-(orig_b + prox_b) / (2*c2[I]*(baseMVA^2) + rho[pg_idx] + tau_prox + rho_prox))))
        u[qg_idx] = max(qgmin[I],
                        min(qgmax[I],
                            (-(l[qg_idx] + rho[qg_idx]*(-x[qg_idx] + z[qg_idx]))) / rho[qg_idx]))
    end
end

function generator_kernel_two_level_proxal_between(
    baseMVA::Float64, ngen::Int, gen_start::Int,
    u::CuDeviceArray{Float64,1}, x::CuDeviceArray{Float64,1},
    z::CuDeviceArray{Float64,1}, l::CuDeviceArray{Float64,1},
    rho::CuDeviceArray{Float64,1},
    pgmin::CuDeviceArray{Float64,1}, pgmax::CuDeviceArray{Float64,1},
    qgmin::CuDeviceArray{Float64,1}, qgmax::CuDeviceArray{Float64,1},
    c2::CuDeviceArray{Float64,1}, c1::CuDeviceArray{Float64,1},
    tau_prox::Float64, rho_prox::Float64,
    pg_ref_prox::CuDeviceArray{Float64,1},
    l_next_prox::CuDeviceArray{Float64,1}, pg_next_prox::CuDeviceArray{Float64,1},
    l_prev_prox::CuDeviceArray{Float64,1}, pg_prev_prox::CuDeviceArray{Float64,1},
    smax_prox::CuDeviceArray{Float64,1}, smin_prox::CuDeviceArray{Float64,1},
    s::CuDeviceArray{Float64,1}
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
        prox_b = -tau_prox*pg_ref_prox[I] + l_next_prox[I] + rho_prox*(-pg_next_prox[I])
        pg_val1 = max(pgmin[I],
                     min(pgmax[I],
                         -(orig_b + prox_b) / (2*c2[I]*(baseMVA)^2 + rho[pg_idx] + tau_prox + rho_prox)))
        s_val1 = -(l_prev_prox[I] + rho_prox*(pg_prev_prox[I] - pg_val1)) / rho_prox

        s_val2 = smax_prox[I]
        prox_b = -tau_prox*pg_ref_prox[I] - (l_prev_prox[I] + rho_prox*(pg_prev_prox[I] + s_val2)) + l_next_prox[I] + rho_prox*(-pg_next_prox[I])
        pg_val2 = max(pgmin[I],
                      min(pgmax[I],
                          (-(orig_b + prox_b) / (2*c2[I]*(baseMVA)^2 + rho[pg_idx] + tau_prox + 2*rho_prox))))
        lg_val2 = l_prev_prox[I] + rho_prox*(pg_prev_prox[I] - pg_val2 + s_val2)

        s_val3 = smin_prox[I]
        prox_b = -tau_prox*pg_ref_prox[I] - (l_prev_prox[I] + rho_prox*(pg_prev_prox[I] + s_val3)) + l_next_prox[I] + rho_prox*(-pg_next_prox[I])
        pg_val3 = max(pgmin[I],
                      min(pgmax[I],
                          (-(orig_b + prox_b) / (2*c2[I]*(baseMVA)^2 + rho[pg_idx] + tau_prox + 2*rho_prox))))
        lg_val3 = l_prev_prox[I] + rho_prox*(pg_prev_prox[I] - pg_val3 + s_val3)

        if smin[I] <= s_val1 && s_val1 <= smax[I]
            u[pg_idx] = pg_val1
            s[I] = s_val1
        elseif lg_val2 <= 0
            u[pg_idx] = pg_val2
            s[I] = s_val2
        elseif lg_val3 >= 0
            u[pg_idx] = pg_val3
            s[I] = s_val3
        else
            # ERROR!
        end
    end
end

function generator_kernel_two_level_proxal_last(
    baseMVA::Float64, ngen::Int, gen_start::Int,
    u::CuDeviceArray{Float64,1}, x::CuDeviceArray{Float64,1},
    z::CuDeviceArray{Float64,1}, l::CuDeviceArray{Float64,1},
    rho::CuDeviceArray{Float64,1},
    pgmin::CuDeviceArray{Float64,1}, pgmax::CuDeviceArray{Float64,1},
    qgmin::CuDeviceArray{Float64,1}, qgmax::CuDeviceArray{Float64,1},
    c2::CuDeviceArray{Float64,1}, c1::CuDeviceArray{Float64,1},
    tau_prox::Float64, rho_prox::Float64,
    pg_ref_prox::CuDeviceArray{Float64,1},
    l_prev_prox::CuDeviceArray{Float64,1}, pg_prev_prox::CuDeviceArray{Float64,1},
    smax_prox::CuDeviceArray{Float64,1}, smin_prox::CuDeviceArray{Float64,1},
    s::CuDeviceArray{Float64,1}
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
        prox_b = -tau_prox*pg_ref_prox[I]
        pg_val1 = max(pgmin[I],
                     min(pgmax[I],
                         -(orig_b + prox_b) / (2*c2[I]*(baseMVA)^2 + rho[pg_idx] + tau_prox)))
        s_val1 = -(l_prev_prox[I] + rho_prox*(pg_prev_prox[I] - pg_val1)) / rho_prox

        s_val2 = smax_prox[I]
        prox_b = -tau_prox*pg_ref_prox[I] - (l_prev_prox[I] + rho_prox*(pg_prev_prox[I] + s_val2))
        pg_val2 = max(pgmin[I],
                      min(pgmax[I],
                          (-(orig_b + prox_b) / (2*c2[I]*(baseMVA)^2 + rho[pg_idx] + tau_prox + rho_prox))))
        lg_val2 = l_prev_prox[I] + rho_prox*(pg_prev_prox[I] - pg_val2 + s_val2)

        s_val3 = smin_prox[I]
        prox_b = -tau_prox*pg_ref_prox[I] - (l_prev_prox[I] + rho_prox*(pg_prev_prox[I] + s_val3))
        pg_val3 = max(pgmin[I],
                      min(pgmax[I],
                          (-(orig_b + prox_b) / (2*c2[I]*(baseMVA)^2 + rho[pg_idx] + tau_prox + rho_prox))))
        lg_val3 = l_prev_prox[I] + rho_prox*(pg_prev_prox[I] - pg_val3 + s_val3)

        if smin[I] <= s_val1 && s_val1 <= smax[I]
            u[pg_idx] = pg_val1
            s[I] = s_val1
        elseif lg_val2 <= 0
            u[pg_idx] = pg_val2
            s[I] = s_val2
        elseif lg_val3 >= 0
            u[pg_idx] = pg_val3
            s[I] = s_val3
        else
            # ERROR!
        end
    end
end

function generator_kernel_two_level(
    gen_mod::ProxALGeneratorModel{CuArray{Float64,1}},
    baseMVA::Float64, u::CuArray{Float64,1}, xbar::CuArray{Float64,1},
    zu::CuArray{Float64,1}, lu::CuArray{Float64,1}, rho_u::CuArray{Float64,1}
)
    nblk = div(gen_mod.ngen, 32, RoundUp)
    if gen_mod.t_curr == 1
        tgpu = CUDA.@timed @cuda threads=32 blocks=nblk generator_kernel_two_level_proxal_first(baseMVA, gen_mod.ngen, gen_mod.gen_start,
                    u, xbar, zu, lu, rho_u, gen_mod.pgmin, gen_mod.pgmax, gen_mod.qgmin, gen_mod.qgmax, gen_mod.c2, gen_mod.c1,
                    gen_mod.tau, gen_mod.rho, gen_mod.pg_ref, gen_mod.l_next, gen_mod.pg_next)
    elseif gen_mod.t_curr < gen_mod.T
        tgpu = CUDA.@timed @cuda threads=32 blocks=nblk generator_kernel_two_level_proxal_between(baseMVA, gen_mod.ngen, gen_mod.gen_start,
                    u, xbar, zu, lu, rho_u, gen_mod.pgmin, gen_mod.pgmax, gen_mod.qgmin, gen_mod.qgmax, gen_mod.c2, gen_mod.c1,
                    gen_mod.tau, gen_mod.rho, gen_mod.pg_ref, gen_mod.l_next, gen_mod.pg_next,
                    gen_mod.l_prev, gen_mod.pg_prev, gen_mod.smax, gen_mod.smin, gen_mod.s_curr)
    else
        tgpu = CUDA.@timed @cuda threads=32 blocks=nblk generator_kernel_two_level_proxal_last(baseMVA, gen_mod.ngen, gen_mod.gen_start,
                    u, xbar, zu, lu, rho_u, gen_mod.pgmin, gen_mod.pgmax, gen_mod.qgmin, gen_mod.qgmax, gen_mod.c2, gen_mod.c1,
                    gen_mod.tau, gen_mod.rho, gen_mod.pg_ref, gen_mod.l_prev, gen_mod.pg_prev,
                    gen_mod.smax, gen_mod.smin, gen_mod.s_curr)
    end

    return tgpu

end

function generator_kernel_two_level_proxal_first(
    baseMVA::Float64, ngen::Int, gen_start::Int,
    u, x, z, l, rho,
    pgmin, pgmax, qgmin, qgmax, c2, c1,
    tau_prox::Float64, rho_prox::Float64, pg_ref_prox,
    l_next_prox, pg_next_prox
)
    for I=1:ngen
        pg_idx = gen_start + 2*(I-1)
        qg_idx = gen_start + 2*(I-1) + 1

        orig_b = c1[I]*baseMVA + l[pg_idx] + rho[pg_idx]*(-x[pg_idx] + z[pg_idx])
        prox_b = -tau_prox*pg_ref_prox[I] + l_next_prox[I] + rho_prox*(-pg_next_prox[I])
        u[pg_idx] = max(pgmin[I],
                        min(pgmax[I],
                            (-(orig_b + prox_b) / (2*c2[I]*(baseMVA^2) + rho[pg_idx] + tau_prox + rho_prox))))
        u[qg_idx] = max(qgmin[I],
                        min(qgmax[I],
                            (-(l[qg_idx] + rho[qg_idx]*(-x[qg_idx] + z[qg_idx]))) / rho[qg_idx]))
    end
end

function generator_kernel_two_level_proxal_between(
    baseMVA::Float64, ngen::Int, gen_start::Int,
    u, x, z, l, rho,
    pgmin, pgmax, qgmin, qgmax, c2, c1,
    tau_prox::Float64, rho_prox::Float64,
    pg_ref_prox, l_next_prox, pg_next_prox, l_prev_prox, pg_prev_prox,
    smax_prox, smin_prox, s
)
    for I=1:ngen
        pg_idx = gen_start + 2*(I-1)
        qg_idx = gen_start + 2*(I-1) + 1

        # Fill out the solution of qg first.
        u[qg_idx] = max(qgmin[I],
                        min(qgmax[I],
                            (-(l[qg_idx] + rho[qg_idx]*(-x[qg_idx] + z[qg_idx]))) / rho[qg_idx]))

        orig_b = c1[I]*baseMVA + l[pg_idx] + rho[pg_idx]*(-x[pg_idx] + z[pg_idx])
        prox_b = -tau_prox*pg_ref_prox[I] + l_next_prox[I] + rho_prox*(-pg_next_prox[I])
        pg_val1 = max(pgmin[I],
                     min(pgmax[I],
                         -(orig_b + prox_b) / (2*c2[I]*(baseMVA)^2 + rho[pg_idx] + tau_prox + rho_prox)))
        s_val1 = -(l_prev_prox[I] + rho_prox*(pg_prev_prox[I] - pg_val1)) / rho_prox

        s_val2 = smax_prox[I]
        prox_b = -tau_prox*pg_ref_prox[I] - (l_prev_prox[I] + rho_prox*(pg_prev_prox[I] + s_val2)) + l_next_prox[I] + rho_prox*(-pg_next_prox[I])
        pg_val2 = max(pgmin[I],
                      min(pgmax[I],
                          (-(orig_b + prox_b) / (2*c2[I]*(baseMVA)^2 + rho[pg_idx] + tau_prox + 2*rho_prox))))
        lg_val2 = l_prev_prox[I] + rho_prox*(pg_prev_prox[I] - pg_val2 + s_val2)

        s_val3 = smin_prox[I]
        prox_b = -tau_prox*pg_ref_prox[I] - (l_prev_prox[I] + rho_prox*(pg_prev_prox[I] + s_val3)) + l_next_prox[I] + rho_prox*(-pg_next_prox[I])
        pg_val3 = max(pgmin[I],
                      min(pgmax[I],
                          (-(orig_b + prox_b) / (2*c2[I]*(baseMVA)^2 + rho[pg_idx] + tau_prox + 2*rho_prox))))
        lg_val3 = l_prev_prox[I] + rho_prox*(pg_prev_prox[I] - pg_val3 + s_val3)

        if smin_prox[I] <= s_val1 && s_val1 <= smax_prox[I]
            u[pg_idx] = pg_val1
            s[I] = s_val1
        elseif lg_val2 <= 0
            u[pg_idx] = pg_val2
            s[I] = s_val2
        elseif lg_val3 >= 0
            u[pg_idx] = pg_val3
            s[I] = s_val3
        else
            @printf("[%d] ERROR! Couldn't find a solution for generator at 1<t<T.\n", I)
            @printf("  pg_val1 = %.6e    s_val1 = %.6e\n", pg_val1, s_val1)
            @printf("  pg_val2 = %.6e    s_val2 = %.6e\n", pg_val2, s_val2)
            @printf("  pg_val3 = %.6e    s_val3 = %.6e\n", pg_val3, s_val3)
            # ERROR!
        end
    end
end

function generator_kernel_two_level_proxal_last(
    baseMVA::Float64, ngen::Int, gen_start::Int,
    u, x, z, l, rho,
    pgmin::Array{Float64,1}, pgmax::Array{Float64,1},
    qgmin::Array{Float64,1}, qgmax::Array{Float64,1},
    c2::Array{Float64,1}, c1::Array{Float64,1},
    tau_prox::Float64, rho_prox::Float64,
    pg_ref_prox, l_prev_prox, pg_prev_prox,
    smax_prox, smin_prox, s
)
    for I=1:ngen
        pg_idx = gen_start + 2*(I-1)
        qg_idx = gen_start + 2*(I-1) + 1

        # Fill out the solution of qg first.
        u[qg_idx] = max(qgmin[I],
                        min(qgmax[I],
                            (-(l[qg_idx] + rho[qg_idx]*(-x[qg_idx] + z[qg_idx]))) / rho[qg_idx]))

        orig_b = c1[I]*baseMVA + l[pg_idx] + rho[pg_idx]*(-x[pg_idx] + z[pg_idx])
        prox_b = -tau_prox*pg_ref_prox[I]
        pg_val1 = max(pgmin[I],
                     min(pgmax[I],
                         -(orig_b + prox_b) / (2*c2[I]*(baseMVA)^2 + rho[pg_idx] + tau_prox)))
        s_val1 = -(l_prev_prox[I] + rho_prox*(pg_prev_prox[I] - pg_val1)) / rho_prox

        s_val2 = smax_prox[I]
        prox_b = -tau_prox*pg_ref_prox[I] - (l_prev_prox[I] + rho_prox*(pg_prev_prox[I] + s_val2))
        pg_val2 = max(pgmin[I],
                      min(pgmax[I],
                          (-(orig_b + prox_b) / (2*c2[I]*(baseMVA)^2 + rho[pg_idx] + tau_prox + rho_prox))))
        lg_val2 = l_prev_prox[I] + rho_prox*(pg_prev_prox[I] - pg_val2 + s_val2)

        s_val3 = smin_prox[I]
        prox_b = -tau_prox*pg_ref_prox[I] - (l_prev_prox[I] + rho_prox*(pg_prev_prox[I] + s_val3))
        pg_val3 = max(pgmin[I],
                      min(pgmax[I],
                          (-(orig_b + prox_b) / (2*c2[I]*(baseMVA)^2 + rho[pg_idx] + tau_prox + rho_prox))))
        lg_val3 = l_prev_prox[I] + rho_prox*(pg_prev_prox[I] - pg_val3 + s_val3)

        if smin_prox[I] <= s_val1 && s_val1 <= smax_prox[I]
            u[pg_idx] = pg_val1
            s[I] = s_val1
        elseif lg_val2 <= 0
            u[pg_idx] = pg_val2
            s[I] = s_val2
        elseif lg_val3 >= 0
            u[pg_idx] = pg_val3
            s[I] = s_val3
        else
            @printf("[%d] ERROR! Couldn't find a solution for generator at t=T.\n", I)
            @printf("  pg_val1 = %.6e    s_val1 = %.6e\n", pg_val1, s_val1)
            @printf("  pg_val2 = %.6e    s_val2 = %.6e\n", pg_val2, s_val2)
            @printf("  pg_val3 = %.6e    s_val3 = %.6e\n", pg_val3, s_val3)
            # ERROR!
        end

#        @printf("[%d]: PG[t-1] = %.6e PG = %.6e QG = %.6e s = %.e\n", I, pg_prev_prox[I], u[pg_idx], u[qg_idx], s[I])
    end
end

function generator_kernel_two_level(
    gen_mod::ProxALGeneratorModel{Array{Float64,1}},
    baseMVA::Float64, u, xbar, zu, lu, rho_u
)
    if gen_mod.t_curr == 1
        tcpu = @timed generator_kernel_two_level_proxal_first(baseMVA, gen_mod.ngen, gen_mod.gen_start,
                    u, xbar, zu, lu, rho_u, gen_mod.pgmin, gen_mod.pgmax, gen_mod.qgmin, gen_mod.qgmax, gen_mod.c2, gen_mod.c1,
                    gen_mod.tau, gen_mod.rho, gen_mod.pg_ref, gen_mod.l_next, gen_mod.pg_next)
    elseif gen_mod.t_curr < gen_mod.T
        tcpu = @timed generator_kernel_two_level_proxal_between(baseMVA, gen_mod.ngen, gen_mod.gen_start,
                    u, xbar, zu, lu, rho_u, gen_mod.pgmin, gen_mod.pgmax, gen_mod.qgmin, gen_mod.qgmax, gen_mod.c2, gen_mod.c1,
                    gen_mod.tau, gen_mod.rho, gen_mod.pg_ref, gen_mod.l_next, gen_mod.pg_next,
                    gen_mod.l_prev, gen_mod.pg_prev, gen_mod.smax, gen_mod.smin, gen_mod.s_curr)
    else
        tcpu = @timed generator_kernel_two_level_proxal_last(baseMVA, gen_mod.ngen, gen_mod.gen_start,
                    u, xbar, zu, lu, rho_u, gen_mod.pgmin, gen_mod.pgmax, gen_mod.qgmin, gen_mod.qgmax, gen_mod.c2, gen_mod.c1,
                    gen_mod.tau, gen_mod.rho, gen_mod.pg_ref, gen_mod.l_prev, gen_mod.pg_prev,
                    gen_mod.smax, gen_mod.smin, gen_mod.s_curr)
    end

    return tcpu
end

