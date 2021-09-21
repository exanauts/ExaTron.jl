function eval_f_generator_kernel_two_level_cpu(I, scale, x, param, c2, c1, baseMVA)
    # x[1] = p_g     : original active power variable
    # x[2] = phat_g  : copied active power for decoupling ramping
    # x[3] = s_g     : slack variable for ramping

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

    f = 0.0
    @inbounds begin
        f += c2[I]*(baseMVA*x[1])^2 + c1[I]*(baseMVA*x[1])
        f += param[1,I]*x[1]
        f += param[2,I]*x[2]
        f += param[3,I]*(x[1] - x[2] - x[3])
        f += param[4,I]*x[1]

        f += 0.5*param[5,I]*(x[1] + param[9,I])^2
        f += 0.5*param[6,I]*(x[2] + param[10,I])^2
        f += 0.5*param[7,I]*(x[1] - x[2] - x[3] + param[11,I])^2
        f += 0.5*param[8,I]*(x[1] + param[12,I])^2
    end

    f *= scale
    return f
end

function eval_g_generator_kernel_two_level_cpu(I, scale, x, g, param, c2, c1, baseMVA)
    # x[1] = p_g     : original active power variable
    # x[2] = phat_g  : copied active power for decoupling ramping
    # x[3] = s_g     : slack variable for ramping

    # param[1,g] : lambda_{p_{t,g}}
    # param[2,g] : lambda_{phat_{t,g}}
    # param[3,g] : lambda_{s_{t,g}}
    # param[4,g] : lambda_{ptilde_{t,g}}
    # param[5,g] : rho_{p_{t,g}}
    # param[6,g] : rho_{phat_{t,g}}
    # param[7,g] : rho_{s_{t,g}}
    # param[8,g] : rho_{ptilde_{t,g}}
    # param[9,g] : -pbar_{t,g} + z_{p_{t,g}
    # param[10,g]: -ptilde_{t-1,g} + z_{phat_{t,g}}
    # param[11,g]: z_{s_{t,g}}
    # param[12,g]: -ptilde_{t,g} + z_{ptilde_{t,g}}

    @inbounds begin
        common = param[3,I] + param[7,I]*(x[1] - x[2] - x[3] + param[11,I])

        g[1] = 2*c2[I]*(baseMVA)^2*x[1] + c1[I]*baseMVA
        g[1] += param[1,I] + param[5,I]*(x[1] + param[9,I])
        g[1] += common
        g[1] += param[4,I] + param[8,I]*(x[1] + param[12,I])
        g[1] *= scale

        g[2] = param[2,I] + param[6,I]*(x[2] + param[10,I])
        g[2] += (-common)
        g[2] *= scale

        g[3] = (-common)
        g[3] *= scale
    end

    return
end

function eval_h_generator_kernel_two_level_cpu(I, x, mode, scale,
    rows, cols, lambda, values, param, c2, c1, baseMVA)

    # x[1] = p_g     : original active power variable
    # x[2] = phat_g  : copied active power for decoupling ramping
    # x[3] = s_g     : slack variable for ramping

    # param[1,g] : lambda_{p_{t,g}}
    # param[2,g] : lambda_{phat_{t,g}}
    # param[3,g] : lambda_{s_{t,g}}
    # param[4,g] : lambda_{ptilde_{t,g}}
    # param[5,g] : rho_{p_{t,g}}
    # param[6,g] : rho_{phat_{t,g}}
    # param[7,g] : rho_{s_{t,g}}
    # param[8,g] : rho_{ptilde_{t,g}}
    # param[9,g] : -pbar_{t,g} + z_{p_{t,g}
    # param[10,g]: -ptilde_{t-1,g} + z_{phat_{t,g}}
    # param[11,g]: z_{s_{t,g}}
    # param[12,g]: -ptilde_{t,g} + z_{ptilde_{t,g}}

    @inbounds begin
        # Sparsity pattern of lower-triangular of Hessian.
        #     1   2   3
        #    ----------
        # 1 | x
        # 2 | x   x
        # 3 | x   x   x
        #    ----------
        if mode == :Structure
            nz = 0
            for j=1:3
                for i=j:3
                    nz += 1
                    rows[nz] = i
                    cols[nz] = j
                end
            end
        else
            nz = 0

            # d2f_dp_{t,g}p_{t,g}
            nz += 1
            values[nz] = scale*(2*c2[I]*(baseMVA)^2 + param[5,I] + param[7,I] + param[8,I])

            # d2f_dp_{t,g}phat_{t,g}
            nz += 1
            values[nz] = scale*(-param[7,I])

            # d2f_dp_{t,g}ds_{t,g}
            nz += 1
            values[nz] = scale*(-param[7,I])

            # d2f_dphat_{t,g}phat_{t,g}
            nz += 1
            values[nz] = scale*(param[6,I] + param[7,I])

            # d2f_dphat_{t,g}s_{t,g}
            nz += 1
            values[nz] = scale*(param[7,I])

            # d2f_ds_{t,g}s_{t,g}
            nz += 1
            values[nz] = scale*(param[7,I])
        end
    end

    return
end

@inline function eval_f_generator_kernel_two_level(
    scale::Float64,
    x::CuDeviceArray{Float64,1}, param::CuDeviceArray{Float64,2},
    c2::Float64, c1::Float64, baseMVA::Float64
)
    # x[1] = p_g     : original active power variable
    # x[2] = phat_g  : copied active power for decoupling ramping
    # x[3] = s_g     : slack variable for ramping

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

    I = blockIdx().x

    f = 0.0
    @inbounds begin
        f += c2*(baseMVA*x[1])^2 + c1*(baseMVA*x[1])
        f += param[1,I]*x[1]
        f += param[2,I]*x[2]
        f += param[3,I]*(x[1] - x[2] - x[3])
        f += param[4,I]*x[1]

        f += 0.5*param[5,I]*(x[1] + param[9,I])^2
        f += 0.5*param[6,I]*(x[2] + param[10,I])^2
        f += 0.5*param[7,I]*(x[1] - x[2] - x[3] + param[11,I])^2
        f += 0.5*param[8,I]*(x[1] + param[12,I])^2
    end

    f *= scale
    CUDA.sync_threads()

    return f
end

@inline function eval_g_generator_kernel_two_level(
    scale::Float64, x::CuDeviceArray{Float64,1}, g::CuDeviceArray{Float64,1},
    param::CuDeviceArray{Float64,2}, c2::Float64, c1::Float64, baseMVA::Float64
)
    # x[1] = p_g     : original active power variable
    # x[2] = phat_g  : copied active power for decoupling ramping
    # x[3] = s_g     : slack variable for ramping

    # param[1,g] : lambda_{p_{t,g}}
    # param[2,g] : lambda_{phat_{t,g}}
    # param[3,g] : lambda_{s_{t,g}}
    # param[4,g] : lambda_{ptilde_{t,g}}
    # param[5,g] : rho_{p_{t,g}}
    # param[6,g] : rho_{phat_{t,g}}
    # param[7,g] : rho_{s_{t,g}}
    # param[8,g] : rho_{ptilde_{t,g}}
    # param[9,g] : -pbar_{t,g} + z_{p_{t,g}
    # param[10,g]: -ptilde_{t-1,g} + z_{phat_{t,g}}
    # param[11,g]: z_{s_{t,g}}
    # param[12,g]: -ptilde_{t,g} + z_{ptilde_{t,g}}

    I = blockIdx().x
    tx = threadIdx().x

    @inbounds begin
        common = param[3,I] + param[7,I]*(x[1] - x[2] - x[3] + param[11,I])

        g1 = 2*c2*(baseMVA)^2*x[1] + c1*baseMVA
        g1 += param[1,I] + param[5,I]*(x[1] + param[9,I])
        g1 += common
        g1 += param[4,I] + param[8,I]*(x[1] + param[12,I])
        g1 *= scale

        g2 = param[2,I] + param[6,I]*(x[2] + param[10,I])
        g2 += (-common)
        g2 *= scale

        g3 = (-common)
        g3 *= scale

        if tx == 1
            g[1] = g1
            g[2] = g2
            g[3] = g3
        end
    end

    CUDA.sync_threads()
    return
end

@inline function eval_h_generator_kernel_two_level(
    scale::Float64, x::CuDeviceArray{Float64,1}, A::CuDeviceArray{Float64,2},
    param::CuDeviceArray{Float64,2}, c2::Float64, c1::Float64, baseMVA::Float64)

    # x[1] = p_g     : original active power variable
    # x[2] = phat_g  : copied active power for decoupling ramping
    # x[3] = s_g     : slack variable for ramping

    # param[1,g] : lambda_{p_{t,g}}
    # param[2,g] : lambda_{phat_{t,g}}
    # param[3,g] : lambda_{s_{t,g}}
    # param[4,g] : lambda_{ptilde_{t,g}}
    # param[5,g] : rho_{p_{t,g}}
    # param[6,g] : rho_{phat_{t,g}}
    # param[7,g] : rho_{s_{t,g}}
    # param[8,g] : rho_{ptilde_{t,g}}
    # param[9,g] : -pbar_{t,g} + z_{p_{t,g}
    # param[10,g]: -ptilde_{t-1,g} + z_{phat_{t,g}}
    # param[11,g]: z_{s_{t,g}}
    # param[12,g]: -ptilde_{t,g} + z_{ptilde_{t,g}}

    I = blockIdx().x
    tx = threadIdx().x

    @inbounds begin
        # Sparsity pattern of lower-triangular of Hessian.
        #     1   2   3
        #    ----------
        # 1 | x
        # 2 | x   x
        # 3 | x   x   x
        #    ----------

        # d2f_dp_{t,g}p_{t,g}
        if tx == 1
            A[1,1] = scale*(2*c2*(baseMVA)^2 + param[5,I] + param[7,I] + param[8,I])

            # d2f_dp_{t,g}phat_{t,g}
            v6 = param[6,I]
            v7 = param[7,I]
            A[2,1] = scale*(-v7)
            A[1,2] = scale*(-v7)

            # d2f_dp_{t,g}ds_{t,g}
            A[3,1] = scale*(-v7)
            A[1,3] = scale*(-v7)

            # d2f_dphat_{t,g}phat_{t,g}
            A[2,2] = scale*(v6 + v7)

            # d2f_dphat_{t,g}s_{t,g}
            A[3,2] = scale*v7
            A[2,3] = scale*v7

            # d2f_ds_{t,g}s_{t,g}
            A[3,3] = scale*v7
        end
    end

    CUDA.sync_threads()

    return
end