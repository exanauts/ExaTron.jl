function generator_kernel_multiperiod_first_cpu(
    baseMVA::Float64, ngen::Int, gen_start::Int,
    u::AbstractVector{Float64}, x::AbstractVector{Float64},
    z::AbstractVector{Float64}, l::AbstractVector{Float64},
    rho::AbstractVector{Float64},
    u_mp::AbstractVector{Float64}, x_mp::AbstractVector{Float64},
    z_mp::AbstractVector{Float64}, l_mp::AbstractVector{Float64},
    rho_mp::AbstractVector{Float64},
    pgmin::Vector{Float64}, pgmax::Vector{Float64},
    qgmin::Vector{Float64}, qgmax::Vector{Float64},
    c2::Vector{Float64}, c1::Vector{Float64}
)
    for I=1:ngen
        pg_idx = gen_start + 2*(I-1)
        qg_idx = gen_start + 2*(I-1) + 1
        pgtilde_idx = 3*I

        u[pg_idx] = max(pgmin[I],
                        min(pgmax[I],
                            (-(c1[I]*baseMVA+l[pg_idx]+rho[pg_idx]*(-x[pg_idx]+z[pg_idx])+
                               l_mp[pgtilde_idx]+rho_mp[pgtilde_idx]*(-x_mp[I]+z_mp[pgtilde_idx])
                               )) / (2*c2[I]*(baseMVA^2)+rho[pg_idx]+rho_mp[pgtilde_idx])))
        u[qg_idx] = max(qgmin[I],
                        min(qgmax[I],
                            (-(l[qg_idx]+rho[qg_idx]*(-x[qg_idx]+z[qg_idx]))) / rho[qg_idx]))
    end

    return
end

function generator_kernel_multiperiod_rest_cpu(
    baseMVA::Float64, ngen::Int, gen_start::Int,
    u::AbstractVector{Float64}, xbar::AbstractVector{Float64},
    z::AbstractVector{Float64}, l::AbstractVector{Float64},
    rho::AbstractVector{Float64},
    u_mp::AbstractVector{Float64},
    xbar_mp::AbstractVector{Float64}, xbar_tm1_mp::AbstractVector{Float64},
    z_mp::AbstractVector{Float64}, l_mp::AbstractVector{Float64},
    rho_mp::AbstractVector{Float64},
    param::Array{Float64,2},
    pgmin::Vector{Float64}, pgmax::Vector{Float64},
    qgmin::Vector{Float64}, qgmax::Vector{Float64},
    c2::Vector{Float64}, c1::Vector{Float64},
    ramp_rate::Vector{Float64}
)
    # x[1] = p_g    : original active power variable
    # x[2] = phat_g : duplicated active power for decoupling ramping
    # x[3] = s_g    : slack variable for ramping

    # param[1,g] : lambda_{p_{t,g}}
    # param[2,g] : lambda_{phat_{t,g}}
    # param[3,g] : lambda_{s_{t,g}}
    # param[4,g] : lambda_{ptilde_{t,g}}
    # param[5,g] : rho_{p_{t,g}}
    # param[6,g] : rho_{phat_{t,g}}
    # param[7,g] : rho_{s_{t,g}}
    # param[8,g] : rho_{ptilde_{t,g}}
    # param[9,g] : -pbar_{t,g} + z_{p_{t,g}}
    # param[10,g]: -ptilde_{t-1,g} + z_{phat_{t,g}}
    # param[11,g]: z_{s_{t,g}}
    # param[12,g]: -ptilde_{t,g} + z_{ptilde_{t,g}}

    x = zeros(3)
    xl = zeros(3)
    xu = zeros(3)

    @inbounds for I=1:ngen
        pg_idx = gen_start + 2*(I-1)
        qg_idx = gen_start + 2*(I-1) + 1
        pg_mp_idx = 3*I - 2

        u[qg_idx] = max(qgmin[I],
                        min(qgmax[I],
                            (-(l[qg_idx]+rho[qg_idx]*(-xbar[qg_idx]+z[qg_idx]))) / rho[qg_idx]))

        xl[1] = xl[2] = pgmin[I]; xu[1] = xu[2] = pgmax[I]
        xl[3] = -ramp_rate[I]; xu[3] = ramp_rate[I]

        x[1] = min(xu[1], max(xl[1], u[gen_start + 2*(I-1)])) # p_{t,g}
        x[2] = min(xu[2], max(xl[2], u_mp[2*I-1]))            # phat_{t,g}
        x[3] = min(xu[3], max(xl[3], u_mp[2*I]))              # s_{t,g}

        param[1,I] = l[pg_idx]
        param[2,I] = l_mp[pg_mp_idx]
        param[3,I] = l_mp[pg_mp_idx+1]
        param[4,I] = l_mp[pg_mp_idx+2]
        param[5,I] = rho[pg_idx]
        param[6,I] = rho_mp[pg_mp_idx]
        param[7,I] = rho_mp[pg_mp_idx+1]
        param[8,I] = rho_mp[pg_mp_idx+2]
        param[9,I] = (-xbar[pg_idx] + z[pg_idx])
        param[10,I] = (-xbar_tm1_mp[I] + z_mp[pg_mp_idx])
        param[11,I] = z_mp[pg_mp_idx+1]
        param[12,I] = (-xbar_mp[I] + z_mp[pg_mp_idx+2])

        function eval_f_gen_cb(x)
            f = eval_f_generator_kernel_two_level_cpu(I, 1.0, x, param, c2, c1, baseMVA)
            return f
        end

        function eval_g_gen_cb(x, g)
            eval_g_generator_kernel_two_level_cpu(I, 1.0, x, g, param, c2, c1, baseMVA)
            return
        end

        function eval_h_gen_cb(x, mode, rows, cols, _scale, lambda, values)
            eval_h_generator_kernel_two_level_cpu(I, x, mode, 1.0, rows, cols, lambda, values, param, c2, c1, baseMVA)
            return
        end

        nele_hess = 6
        tron = ExaTron.createProblem(3, xl, xu, nele_hess, eval_f_gen_cb, eval_g_gen_cb, eval_h_gen_cb;
                                    :tol => 1e-6, :matrix_type => :Dense, :max_minor => 200,
                                    :frtol => 1e-12)
        tron.x .= x
        status = ExaTron.solveProblem(tron)
        x .= tron.x

        u[pg_idx] = x[1]
        u_mp[2*I-1] = x[2]
        u_mp[2*I] = x[3]
#=
        println("I = ", I)
        println(" status = ", status, " iter = ", tron.minor_iter)
        println(" x = ", x)
=#
    end
end

#=
function generator_kernel_two_level_multiperiod(
    baseMVA::Float64, ngen::Int, gen_start::Int,
    u::CuDeviceArray{Float64,1}, x::CuDeviceArray{Float64,1}, z::CuDeviceArray{Float64,1},
    l::CuDeviceArray{Float64,1}, rho::CuDeviceArray{Float64,1},
    pgmin::CuDeviceArray{Float64,1}, pgmax::CuDeviceArray{Float64,1},
    qgmin::CuDeviceArray{Float64,1}, qgmax::CuDeviceArray{Float64,1},
    c2::CuDeviceArray{Float64,1}, c1::CuDeviceArray{Float64,1}
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

function generator_kernel_two_level_multiperiod_phat(baseMVA, sval, pg_idx, I, sI,
    x, z, l, rho, mp_x, mp_z, mp_l, mp_rho,
    pgmin::Array{Float64,1}, pgmax::Array{Float64,1},
    c2::Array{Float64,1}, c1::Array{Float64,1}
)
    # Three cases: i) l_g < phat_g < u_g; ii) phat_g = u_g; and iii) phat_g = l_g.
    #
    # Case i) l_g < phat_g < u_g:
    a = 2*c2[I]*(baseMVA)^2 + rho[pg_idx] + mp_rho[sI] - (mp_rho[sI]^2)/(mp_rho[I] + mp_rho[sI])
    b = c1[I]*baseMVA + l[pg_idx] + rho[pg_idx]*(-x[pg_idx] + z[pg_idx]) + mp_l[sI] +
        mp_rho[sI]*((mp_l[I] + mp_rho[I]*(-mp_x[I] + mp_z[I]) - (mp_l[sI] + mp_rho[sI]*(-sval + mp_z[sI]))) / (mp_rho[I] + mp_rho[sI])
                    - sval + mp_z[sI])
    p_val1 = max(pgmin[I], min(pgmax[I], (-b) / (a)))
    phat_val1 = (-(mp_l[I] + mp_rho[I]*(-mp_x[I] + mp_z[I])) + (mp_l[sI] + mp_rho[sI]*(p_val1 - sval + mp_z[sI]))) / (mp_rho[I] + mp_rho[sI])

    # Case ii) phat_g = u_g:
    phat_val2 = pgmax[I]
    a = 2*c2[I]*(baseMVA)^2 + rho[pg_idx] + mp_rho[sI]
    b = c1[I]*baseMVA + l[pg_idx] + rho[pg_idx](-x[pg_idx] + z[pg_idx] ) +
        mp_l[sI] + mp_rho[sI]*(-phat_val2 - sval + mp_z[sI])
    p_val2 = max(pgmin[I], min(pgmax[I], (-b) / (a)))
    lg_val2 = mp_l[I] + mp_rho[I]*(phat_val2 - mp_x[I] + mp_z[I]) -
              (mp_l[sI] + mp_rho[sI]*(p_val2 - phat_val2 - sval + mp_z[sI]))

    # Case iii) phat_g = l_g:
    phat_val3 = pgmin[I]
    a = 2*c2[I]*(baseMVA)^2 + rho[pg_idx] + mp_rho[sI]
    b = c1[I]*baseMVA + l[pg_idx] + rho[pg_idx](-x[pg_idx] + z[pg_idx] ) +
        mp_l[sI] + mp_rho[sI]*(-phat_val3 - sval + mp_z[sI])
    p_val3 = max(pgmin[I], min(pgmax[I], (-b) / (a)))
    lg_val3 = mp_l[I] + mp_rho[I]*(phat_val3 - mp_x[I] + mp_z[I]) -
              (mp_l[sI] + mp_rho[sI]*(p_val3 - phat_val3 - sval + mp_z[sI]))

    if pgmin[I] <= phat_val1 && phat_val1 <= pgmax[I]
        return p_val1, phat_val1
    elseif lg_val2 <= 0
        return p_val2, phat_val2
    elseif lg_val3 >= 0
        return p_val3, phat_val3
    else
        error("Error: generator_kernel: this can't happen.")
    end
end

function generator_kernel_two_level_multiperiod(
    baseMVA::Float64, ngen::Int, gen_start::Int,
    u, x, z, l, rho, mp_u, mp_x, mp_z, mp_l, mp_rho,
    pgmin::Array{Float64,1}, pgmax::Array{Float64,1},
    qgmin::Array{Float64,1}, qgmax::Array{Float64,1},
    c2::Array{Float64,1}, c1::Array{Float64,1},
    smin::Array{Float64,1}, smax::Array{Float64,1}
)
    for I=1:ngen
        pg_idx = gen_start + 2*(I-1)
        qg_idx = gen_start + 2*(I-1) + 1
        sI = ngen + I

        # Fill out the solution for qg first.
        u[qg_idx] = max(qgmin[I],
                        min(qgmax[I],
                            (-(l[qg_idx] + rho[qg_idx]*(-x[qg_idx] + z[qg_idx]))) / rho[qg_idx]))

        # Three cases: i) l_s < s < u_s; ii) s = u_s; and iii) s = l_s.
        #
        # Case i) l_s < s < u_s:

        a = 2*c2[I]*(baseMVA)^2 + rho[pg_idx]
        b = c1[I]*baseMVA + l[pg_idx] + rho[pg_idx]*(-x[pg_idx] + z[pg_idx])
        p_val1 = max(pgmin[I], min(pgmax[I], (-b)/a))

        b = mp_l[I] + mp_rho[I]*(-mp_x[I] + mp_z[I])
        phat_val1 = max(pgmin[I], min(pgmax[I], (-b) / (mp_rho[I])))
        s_val1 = (mp_l[sI] + mp_rho[sI]*(p_val1 - phat_val1 + mp_z[sI])) / (mp_rho[sI])

        # Case ii) s = u_s:
        s_val2 = smax[I]
        p_val2, phat_val2 = generator_kernel_two_level_multiperiod_phat(baseMVA, s_val2, pg_idx, I, sI,
                                x, z, l, rho, mp_x, mp_z, mp_l, mp_rho, pgmin, pgmax, c2, c1)
        lg_val2 = -(mp_l[sI] + mp_rho[sI]*(p_val2 - phat_val2 - s_val2 + mp_z[sI]))

        # Case iii) s = l_s:
        s_val3 = smin[I]
        p_val3, phat_val3 = generator_kernel_two_level_multiperiod_phat(baseMVA, s_val3, pg_idx, I, sI,
                                x, z, l, rho, mp_x, mp_z, mp_l, mp_rho, pgmin, pgmax, c2, c1)
        lg_val3 = -(mp_l[sI] + mp_rho[sI]*(p_val3 - phat_val3 - s_val3 + mp_z[sI]))

        if smin[I] <= s_val1 && s_val1 <= smax[I]
            u[pg_idx] = p_val1
            mp_u[I] = phat_val1
            mp_u[sI] = s_val1
        elseif lg_val2 <= 0
            u[pg_idx] = p_val2
            mp_u[I] = phat_val2
            mp_u[sI] = s_val2
        elseif lg_val3 >= 0
            u[pg_idx] = p_val3
            mp_u[I] = phat_val3
            mp_u[sI] = s_val3
        else
            error("Error in brute-force.")
        end
    end

    return
end

function generator_kernel_two_level_multiperiod(
    model::ModelMultiperiod{Float64,CuArray{Float64,1},CuArray{Int,1}},
    baseMVA::Float64, u::CuArray{Float64,1}, xbar::CuArray{Float64,1},
    zu::CuArray{Float64,1}, lu::CuArray{Float64,1}, rho_u::CuArray{Float64,1}
)
    nblk = div(model.ngen, 32, RoundUp)
    tgpu = CUDA.@timed @cuda threads=32 blocks=nblk generator_kernel_two_level_multiperiod(baseMVA, model.ngen, model.gen_start,
                u, xbar, zu, lu, rho_u, model.pgmin, model.pgmax, model.qgmin, model.qgmax, model.c2, model.c1)
    return tgpu
end

function generator_kernel_two_level_multiperiod(
    model::ModelMultiperiod{Float64,Array{Float64,1},Array{Int,1}},
    baseMVA::Float64, u, xbar, zu, lu, rho_u
)
    sol = model.solution

    if model.t_start == 1
        tcpu = @timed generator_kernel_two_level(baseMVA, model.ngen, model.gen_start,
                u, xbar, zu, lu, rho_u, model.pgmin, model.pgmax, model.qgmin, model.qgmax, model.c2, model.c1)
    else
        tcpu = @timed generator_kernel_two_level_multiperiod(baseMVA, model.ngen, model.gen_start,
                u, xbar, zu, lu, rho_u, sol.mp_x_curr, sol.mp_xbar_curr, sol.mp_z_curr, sol.mp_l_curr, sol.mp_rho,
                model.pgmin, model.pgmax, model.qgmin, model.qgmax, model.c2, model.c1,
                model.smin, model.smax)
    end

    return tcpu
end
=#