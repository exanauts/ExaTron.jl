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
    return
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
            # ERROR!
        end
    end
    return
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
            # ERROR!
        end
    end
    return
end

#=
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
=#

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

function generator_kernel_two_level_proxal(ngen::Int, gen_start::Int,
    u::CuDeviceArray{Float64,1}, xbar::CuDeviceArray{Float64,1}, z::CuDeviceArray{Float64,1},
    l::CuDeviceArray{Float64,1}, rho::CuDeviceArray{Float64,1},
    pgmin::CuDeviceArray{Float64,1}, pgmax::CuDeviceArray{Float64,1},
    qgmin::CuDeviceArray{Float64,1}, qgmax::CuDeviceArray{Float64,1},
    smin::CuDeviceArray{Float64,1}, smax::CuDeviceArray{Float64,1}, s::CuDeviceArray{Float64,1},
    _A::CuDeviceArray{Float64,1}, _c::CuDeviceArray{Float64,1})

    tx = threadIdx().x
    I = blockIdx().x

    if I <= ngen
        n = 2
        x = @cuDynamicSharedMem(Float64, n)
        xl = @cuDynamicSharedMem(Float64, n, n*sizeof(Float64))
        xu = @cuDynamicSharedMem(Float64, n, (2*n)*sizeof(Float64))

        A = @cuDynamicSharedMem(Float64, (n,n), (13*n+3)*sizeof(Float64)+(3*n+3)*sizeof(Int))
        c = @cuDynamicSharedMem(Float64, n, (13*n+3+3*n^2)*sizeof(Float64)+(3*n+3)*sizeof(Int))

        pg_idx = gen_start + 2*(I-1)
        qg_idx = gen_start + 2*(I-1) + 1

        u[qg_idx] = max(qgmin[I],
                        min(qgmax[I],
                            (-(l[qg_idx] + rho[qg_idx]*(-xbar[qg_idx] + z[qg_idx]))) / rho[qg_idx]))

        A_start = 4*(I-1)
        c_start = 2*(I-1)
        if tx <= n
            @inbounds begin
                for j=1:n
                    A[tx,j] = _A[n*(j-1)+tx + A_start]
                end
                c[tx] = _c[tx + c_start]

                if tx == 1
                    A[1,1] += rho[pg_idx]
                    c[1] += l[pg_idx] + rho[pg_idx]*(-xbar[pg_idx] + z[pg_idx])
                end
            end
        end
        CUDA.sync_threads()

        @inbounds begin
            xl[1] = pgmin[I]
            xu[1] = pgmax[I]
            xl[2] = smin[I]
            xu[2] = smax[I]
            x[1] = min(xu[1], max(xl[1], u[pg_idx]))
            x[2] = min(xu[2], max(xl[2], s[I]))
            CUDA.sync_threads()

            status, minor_iter = tron_qp_kernel(n, 500, 200, 1e-6, 1.0, x, xl, xu, A, c)

            u[pg_idx] = x[1]
            s[I] = x[2]
        end
    end

    return
end

function generator_kernel_two_level_proxal(ngen::Int, gen_start::Int,
    u, xbar, z, l, rho,
    pgmin::Array{Float64,1}, pgmax::Array{Float64,1}, qgmin::Array{Float64,1}, qgmax::Array{Float64,1},
    smin::Array{Float64,1}, smax::Array{Float64,1}, s::Array{Float64,1},
    _A::Array{Float64,1}, _c::Array{Float64,1})

    n = 2
    x = zeros(n)
    xl = zeros(n)
    xu = zeros(n)
    A = zeros(n,n)
    c = zeros(n)

    @inbounds for I=1:ngen
        pg_idx = gen_start + 2*(I-1)
        qg_idx = gen_start + 2*(I-1) + 1

        u[qg_idx] = max(qgmin[I],
                        min(qgmax[I],
                            (-(l[qg_idx] + rho[qg_idx]*(-xbar[qg_idx] + z[qg_idx]))) / rho[qg_idx]))

        A_start = 4*(I-1)
        c_start = 2*(I-1)
        for i=1:n
            for j=1:n
                A[i,j] = _A[n*(j-1)+i + A_start]
            end
            c[i] = _c[i + c_start]
        end

        A[1,1] += rho[pg_idx]
        c[1] += l[pg_idx] + rho[pg_idx]*(-xbar[pg_idx] + z[pg_idx])

        xl[1] = pgmin[I]
        xu[1] = pgmax[I]
        xl[2] = smin[I]
        xu[2] = smax[I]
        x[1] = min(xu[1], max(xl[1], u[pg_idx]))
        x[2] = min(xu[2], max(xl[2], s[I]))

        function eval_f_cb(x)
            f = 0.0
            for j=1:n
                for i=1:n
                    f += x[i]*A[i,j]*x[j]
                end
            end
            f *= 0.5
            for i=1:n
                f += c[i]*x[i]
            end
            return f
        end

        function eval_g_cb(x, g)
            for i=1:n
                g[i] = 0.0
                for j=1:n
                    g[i] += A[i,j]*x[j]
                end
                g[i] += c[i]
            end
            return
        end

        function eval_h_cb(x, mode, rows, cols, _scale, lambda, values)
            if mode == :Structure
                rows[1] = 1; cols[1] = 1
                rows[2] = 2; cols[2] = 1
                rows[3] = 2; cols[3] = 2
            else
                values[1] = A[1,1]
                values[2] = A[2,1]
                values[3] = A[2,2]
            end
            return
        end

        nele_hess = 3
        tron = ExaTron.createProblem(n, xl, xu, nele_hess, eval_f_cb, eval_g_cb, eval_h_cb;
                                     :tol => 1e-6, :matrix_type => :Dense, :max_minor => 200,
                                     :frtol => 1e-12)

        tron.x .= x
        status = ExaTron.solveProblem(tron)

        u[pg_idx] = tron.x[1]
        s[I] = tron.x[2]
    end

    return
end

function generator_kernel_two_level(
    gen_mod::ProxALGeneratorModel{CuArray{Float64,1}},
    baseMVA::Float64, u::CuArray{Float64,1}, xbar::CuArray{Float64,1},
    zu::CuArray{Float64,1}, lu::CuArray{Float64,1}, rho_u::CuArray{Float64,1})

    n = 2
    shmem_size = sizeof(Float64)*(14*n+3+3*n^2) + sizeof(Int)*(3*n+3)

    tgpu = CUDA.@timed @cuda threads=32 blocks=gen_mod.ngen shmem=shmem_size generator_kernel_two_level_proxal(
                gen_mod.ngen, gen_mod.gen_start,
                u, xbar, zu, lu, rho_u, gen_mod.pgmin, gen_mod.pgmax, gen_mod.qgmin, gen_mod.qgmax,
                gen_mod.smin, gen_mod.smax, gen_mod.s_curr, gen_mod.Q, gen_mod.c)
    return tgpu
end

function generator_kernel_two_level(
    gen_mod::ProxALGeneratorModel{Array{Float64,1}},
    baseMVA::Float64, u, xbar, zu, lu, rho_u)
    tcpu = @timed generator_kernel_two_level_proxal(gen_mod.ngen, gen_mod.gen_start,
                u, xbar, zu, lu, rho_u, gen_mod.pgmin, gen_mod.pgmax, gen_mod.qgmin, gen_mod.qgmax,
                gen_mod.smin, gen_mod.smax, gen_mod.s_curr, gen_mod.Q, gen_mod.c)
    return tcpu
end

#=
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
=#
