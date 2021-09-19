function auglag_linelimit_kernel_two_level(
    n::Int, nline::Int, line_start::Int, bus_start::Int,
    major_iter::Int, max_auglag::Int, mu_max::Float64, scale::Float64,
    u::CuDeviceArray{Float64}, xbar::CuDeviceArray{Float64},
    z::CuDeviceArray{Float64}, l::CuDeviceArray{Float64}, rho::CuDeviceArray{Float64},
    shift_lines::Int, param::CuDeviceArray{Float64},
    _YffR::CuDeviceArray{Float64}, _YffI::CuDeviceArray{Float64},
    _YftR::CuDeviceArray{Float64}, _YftI::CuDeviceArray{Float64},
    _YttR::CuDeviceArray{Float64}, _YttI::CuDeviceArray{Float64},
    _YtfR::CuDeviceArray{Float64}, _YtfI::CuDeviceArray{Float64},
    frBound::CuDeviceArray{Float64}, toBound::CuDeviceArray{Float64},
    brBusIdx::CuDeviceArray{Int})

    tx = threadIdx().x
    I = blockIdx().x
    id_line = I + shift_lines

    x = @cuDynamicSharedMem(Float64, n)
    xl = @cuDynamicSharedMem(Float64, n, n*sizeof(Float64))
    xu = @cuDynamicSharedMem(Float64, n, (2*n)*sizeof(Float64))

    @inbounds begin
        YffR = _YffR[id_line]; YffI = _YffI[id_line]
        YftR = _YftR[id_line]; YftI = _YftI[id_line]
        YttR = _YttR[id_line]; YttI = _YttI[id_line]
        YtfR = _YtfR[id_line]; YtfI = _YtfI[id_line]

        pij_idx = line_start + 8*(I-1)
        xbar_pij_idx = line_start + 4*(I-1)

        xl[1] = sqrt(frBound[2*(id_line-1)+1])
        xu[1] = sqrt(frBound[2*id_line])
        xl[2] = sqrt(toBound[2*(id_line-1)+1])
        xu[2] = sqrt(toBound[2*id_line])
        xl[3] = -2*pi
        xu[3] = 2*pi
        xl[4] = -2*pi
        xu[4] = 2*pi
        xl[5] = -param[29,id_line]
        xu[5] = 0.0
        xl[6] = -param[29,id_line]
        xu[6] = 0.0

        x[1] = min(xu[1], max(xl[1], sqrt(u[pij_idx+4])))
        x[2] = min(xu[2], max(xl[2], sqrt(u[pij_idx+5])))
        x[3] = min(xu[3], max(xl[3], u[pij_idx+6]))
        x[4] = min(xu[4], max(xl[4], u[pij_idx+7]))
        x[5] = min(xu[5], max(xl[5], -(u[pij_idx]^2 + u[pij_idx+1]^2)))
        x[6] = min(xu[6], max(xl[6], -(u[pij_idx+2]^2 + u[pij_idx+3]^2)))

        fr_idx = bus_start + 2*(brBusIdx[2*(id_line-1)+1] - 1)
        to_idx = bus_start + 2*(brBusIdx[2*id_line] - 1)

        param[1,id_line] = l[pij_idx]
        param[2,id_line] = l[pij_idx+1]
        param[3,id_line] = l[pij_idx+2]
        param[4,id_line] = l[pij_idx+3]
        param[5,id_line] = l[pij_idx+4]
        param[6,id_line] = l[pij_idx+5]
        param[7,id_line] = l[pij_idx+6]
        param[8,id_line] = l[pij_idx+7]
        param[9,id_line] = rho[pij_idx]
        param[10,id_line] = rho[pij_idx+1]
        param[11,id_line] = rho[pij_idx+2]
        param[12,id_line] = rho[pij_idx+3]
        param[13,id_line] = rho[pij_idx+4]
        param[14,id_line] = rho[pij_idx+5]
        param[15,id_line] = rho[pij_idx+6]
        param[16,id_line] = rho[pij_idx+7]
        param[17,id_line] = xbar[xbar_pij_idx] - z[pij_idx]
        param[18,id_line] = xbar[xbar_pij_idx+1] - z[pij_idx+1]
        param[19,id_line] = xbar[xbar_pij_idx+2] - z[pij_idx+2]
        param[20,id_line] = xbar[xbar_pij_idx+3] - z[pij_idx+3]
        param[21,id_line] = xbar[fr_idx] - z[pij_idx+4]
        param[22,id_line] = xbar[to_idx] - z[pij_idx+5]
        param[23,id_line] = xbar[fr_idx+1] - z[pij_idx+6]
        param[24,id_line] = xbar[to_idx+1] - z[pij_idx+7]

        if major_iter == 1
            param[27,id_line] = 10.0
            mu = 10.0
        else
            mu = param[27,id_line]
        end

        CUDA.sync_threads()

        eta = 1.0 / mu^0.1
        omega = 1.0 / mu

        it = 0
        terminate = false

        while !terminate
            it += 1

            # Solve the branch problem.
            status, minor_iter = tron_linelimit_kernel(n, 0, 500, 200, 1e-6, scale, true, x, xl, xu,
                                                       param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI)

            # Check the termination condition.
            vi_vj_cos = x[1]*x[2]*cos(x[3] - x[4])
            vi_vj_sin = x[1]*x[2]*sin(x[3] - x[4])
            pij = YffR*x[1]^2 + YftR*vi_vj_cos + YftI*vi_vj_sin
            qij = -YffI*x[1]^2 - YftI*vi_vj_cos + YftR*vi_vj_sin
            pji = YttR*x[2]^2 + YtfR*vi_vj_cos - YtfI*vi_vj_sin
            qji = -YttI*x[2]^2 - YtfI*vi_vj_cos - YtfR*vi_vj_sin

            cviol1 = pij^2 + qij^2 + x[5]
            cviol2 = pji^2 + qji^2 + x[6]

            cnorm = max(abs(cviol1), abs(cviol2))

            if cnorm <= eta
                if cnorm <= 1e-6
                    terminate = true
                else
                    if tx == 1
                        param[25,id_line] += mu*cviol1
                        param[26,id_line] += mu*cviol2
                    end
                    eta = eta / mu^0.9
                    omega  = omega / mu
                end
            else
                mu = min(mu_max, mu*10)
                eta = 1 / mu^0.1
                omega = 1 / mu
                param[27,id_line] = mu
            end

            if it >= max_auglag
                terminate = true
            end

            CUDA.sync_threads()
        end

        vi_vj_cos = x[1]*x[2]*cos(x[3] - x[4])
        vi_vj_sin = x[1]*x[2]*sin(x[3] - x[4])
        u[pij_idx] = YffR*x[1]^2 + YftR*vi_vj_cos + YftI*vi_vj_sin
        u[pij_idx+1] = -YffI*x[1]^2 - YftI*vi_vj_cos + YftR*vi_vj_sin
        u[pij_idx+2] = YttR*x[2]^2 + YtfR*vi_vj_cos - YtfI*vi_vj_sin
        u[pij_idx+3] = -YttI*x[2]^2 - YtfI*vi_vj_cos - YtfR*vi_vj_sin
        u[pij_idx+4] = x[1]^2
        u[pij_idx+5] = x[2]^2
        u[pij_idx+6] = x[3]
        u[pij_idx+7] = x[4]
        param[27,id_line] = mu

        CUDA.sync_threads()
    end

    return
end

function auglag_linelimit_kernel_two_level_cpu(
    n::Int, nline::Int, line_start::Int, bus_start::Int,
    major_iter::Int, max_auglag::Int, mu_max::Float64, scale::Float64,
    u, xbar, z, l, rho, shift_lines::Int, param,
    _YffR, _YffI, _YftR, _YftI, _YttR, _YttI, _YtfR, _YtfI,
    frBound, toBound, brBusIdx)

    avg_auglag_it = 0
    avg_minor_it = 0

    x = zeros(n)
    xl = zeros(n)
    xu = zeros(n)

    @inbounds for I=1:nline
        YffR = _YffR[I]; YffI = _YffI[I]
        YftR = _YftR[I]; YftI = _YftI[I]
        YttR = _YttR[I]; YttI = _YttI[I]
        YtfR = _YtfR[I]; YtfI = _YtfI[I]

        pij_idx = line_start + 8*(I-1)
        xbar_pij_idx = line_start + 4*(I-1)

        xl[1] = sqrt(frBound[2*(I-1)+1])
        xu[1] = sqrt(frBound[2*I])
        xl[2] = sqrt(toBound[2*(I-1)+1])
        xu[2] = sqrt(toBound[2*I])
        xl[3] = -2*pi
        xu[3] = 2*pi
        xl[4] = -2*pi
        xu[4] = 2*pi
        xl[5] = -param[29,I]
        xu[5] = 0.0
        xl[6] = -param[29,I]
        xu[6] = 0.0

        x[1] = min(xu[1], max(xl[1], sqrt(u[pij_idx+4])))
        x[2] = min(xu[2], max(xl[2], sqrt(u[pij_idx+5])))
        x[3] = min(xu[3], max(xl[3], u[pij_idx+6]))
        x[4] = min(xu[4], max(xl[4], u[pij_idx+7]))
        x[5] = min(xu[5], max(xl[5], -(u[pij_idx]^2 + u[pij_idx+1]^2)))
        x[6] = min(xu[6], max(xl[6], -(u[pij_idx+2]^2 + u[pij_idx+3]^2)))

        fr_idx = bus_start + 2*(brBusIdx[2*(I-1)+1] - 1)
        to_idx = bus_start + 2*(brBusIdx[2*I] - 1)

        param[1,I] = l[pij_idx]
        param[2,I] = l[pij_idx+1]
        param[3,I] = l[pij_idx+2]
        param[4,I] = l[pij_idx+3]
        param[5,I] = l[pij_idx+4]
        param[6,I] = l[pij_idx+5]
        param[7,I] = l[pij_idx+6]
        param[8,I] = l[pij_idx+7]
        param[9,I] = rho[pij_idx]
        param[10,I] = rho[pij_idx+1]
        param[11,I] = rho[pij_idx+2]
        param[12,I] = rho[pij_idx+3]
        param[13,I] = rho[pij_idx+4]
        param[14,I] = rho[pij_idx+5]
        param[15,I] = rho[pij_idx+6]
        param[16,I] = rho[pij_idx+7]
        param[17,I] = xbar[xbar_pij_idx] - z[pij_idx]
        param[18,I] = xbar[xbar_pij_idx+1] - z[pij_idx+1]
        param[19,I] = xbar[xbar_pij_idx+2] - z[pij_idx+2]
        param[20,I] = xbar[xbar_pij_idx+3] - z[pij_idx+3]
        param[21,I] = xbar[fr_idx] - z[pij_idx+4]
        param[22,I] = xbar[to_idx] - z[pij_idx+5]
        param[23,I] = xbar[fr_idx+1] - z[pij_idx+6]
        param[24,I] = xbar[to_idx+1] - z[pij_idx+7]

        if major_iter == 1
            param[27,I] = 10.0
            mu = 10.0
        else
            mu = param[27,I]
        end

        function eval_f_cb(x)
            f = eval_f_polar_linelimit_kernel_cpu(I, x, param, scale, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI)
            return f
        end

        function eval_g_cb(x, g)
            eval_grad_f_polar_linelimit_kernel_cpu(I, x, g, param, scale, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI)
            return
        end

        function eval_h_cb(x, mode, rows, cols, _scale, lambda, values)
            eval_h_polar_linelimit_kernel_cpu(I, x, mode, rows, cols, lambda, values, param,
                              scale, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI)
            return
        end

        eta = 1 / mu^0.1
        omega = 1 / mu
        max_feval = 500
        max_minor = 100
        gtol = 1e-6

        nele_hess = 20
        tron = ExaTron.createProblem(n, xl, xu, nele_hess, eval_f_cb, eval_g_cb, eval_h_cb;
                                     :tol => gtol, :matrix_type => :Dense, :max_minor => 200,
                                     :frtol => 1e-12)
        tron.x .= x
        it = 0
        avg_tron_minor = 0
        terminate = false

        while !terminate
            it += 1

            # Solve the branch problem.
            status = ExaTron.solveProblem(tron)
            if status != :Solve_Succeeded
                println("Solve failed for branch ", I, " with status = ", status)
            end
            x .= tron.x
            avg_tron_minor += tron.minor_iter

            # Check the termination condition.
            vi_vj_cos = x[1]*x[2]*cos(x[3] - x[4])
            vi_vj_sin = x[1]*x[2]*sin(x[3] - x[4])
            pij = YffR*x[1]^2 + YftR*vi_vj_cos + YftI*vi_vj_sin
            qij = -YffI*x[1]^2 - YftI*vi_vj_cos + YftR*vi_vj_sin
            pji = YttR*x[2]^2 + YtfR*vi_vj_cos - YtfI*vi_vj_sin
            qji = -YttI*x[2]^2 - YtfI*vi_vj_cos - YtfR*vi_vj_sin

            cviol1 = pij^2 + qij^2 + x[5]
            cviol2 = pji^2 + qji^2 + x[6]

            cnorm = max(abs(cviol1), abs(cviol2))
            #=
            @printf("I = %d cnorm = %.6e mu = %.6e pij^2 + qij^2 = %.6e, x[5] = %.6e pji^2 + qji^2 = %.6e x[6] = %.6e\n",
                     I, cnorm, mu, pij^2 + qij^2, x[5], pji^2 + qji^2, x[6])
            =#

            if cnorm <= eta
                if cnorm <= 1e-6
                    terminate = true
                else
                    param[25,I] += mu*cviol1
                    param[26,I] += mu*cviol2

                    eta = eta / mu^0.9
                    omega  = omega / mu
                end
            else
                mu = min(mu_max, mu*10)
                eta = 1 / mu^0.1
                omega = 1 / mu
                param[27,I] = mu
            end

            if it >= max_auglag
                terminate = true
            end
        end

        avg_auglag_it += it
        avg_minor_it += (avg_tron_minor / it)
        vi_vj_cos = x[1]*x[2]*cos(x[3] - x[4])
        vi_vj_sin = x[1]*x[2]*sin(x[3] - x[4])
        u[pij_idx] = YffR*x[1]^2 + YftR*vi_vj_cos + YftI*vi_vj_sin
        u[pij_idx+1] = -YffI*x[1]^2 - YftI*vi_vj_cos + YftR*vi_vj_sin
        u[pij_idx+2] = YttR*x[2]^2 + YtfR*vi_vj_cos - YtfI*vi_vj_sin
        u[pij_idx+3] = -YttI*x[2]^2 - YtfI*vi_vj_cos - YtfR*vi_vj_sin
        u[pij_idx+4] = x[1]^2
        u[pij_idx+5] = x[2]^2
        u[pij_idx+6] = x[3]
        u[pij_idx+7] = x[4]
        param[27,I] = mu
    end

    return (avg_auglag_it / nline), (avg_minor_it / nline)
end

#=
function auglag_linelimit_kernel_ipopt_two_level_cpu(
    n::Int, nline::Int, line_start::Int, bus_start::Int,
    major_iter::Int, max_auglag::Int, mu_max::Float64, scale::Float64,
    u, xbar, z, l, rho, shift_lines::Int, param,
    _YffR, _YffI, _YftR, _YftI, _YttR, _YttI, _YtfR, _YtfI,
    frBound, toBound, brBusIdx)

    avg_auglag_it = 0
    avg_minor_it = 0

    x_val = zeros(n)

    @inbounds for I=1:nline
        YffR = _YffR[I]; YffI = _YffI[I]
        YftR = _YftR[I]; YftI = _YftI[I]
        YttR = _YttR[I]; YttI = _YttI[I]
        YtfR = _YtfR[I]; YtfI = _YtfI[I]

        pij_idx = line_start + 8*(I-1)
        xbar_pij_idx = line_start + 4*(I-1)

        jm = JuMP.Model(Ipopt.Optimizer)

        @variable(jm, x[1:n])
        JuMP.set_lower_bound(x[1], sqrt(frBound[2*(I-1)+1]))
        JuMP.set_upper_bound(x[1], sqrt(frBound[2*I]))
        JuMP.set_lower_bound(x[2], sqrt(toBound[2*(I-1)+1]))
        JuMP.set_upper_bound(x[2], sqrt(toBound[2*I]))
        JuMP.set_lower_bound(x[3], -2*pi)
        JuMP.set_upper_bound(x[3], 2*pi)
        JuMP.set_lower_bound(x[4], -2*pi)
        JuMP.set_upper_bound(x[4], 2*pi)
        JuMP.set_lower_bound(x[5], -param[29,I])
        JuMP.set_upper_bound(x[5], 0.0)
        JuMP.set_lower_bound(x[6], -param[29,I])
        JuMP.set_upper_bound(x[6], 0.0)
        JuMP.set_start_value(x[1], min(JuMP.upper_bound(x[1]), max(JuMP.lower_bound(x[1]), sqrt(u[pij_idx+4]))))
        JuMP.set_start_value(x[2], min(JuMP.upper_bound(x[2]), max(JuMP.lower_bound(x[2]), sqrt(u[pij_idx+5]))))
        JuMP.set_start_value(x[3], min(JuMP.upper_bound(x[3]), max(JuMP.lower_bound(x[3]), u[pij_idx+6])))
        JuMP.set_start_value(x[4], min(JuMP.upper_bound(x[4]), max(JuMP.lower_bound(x[4]), u[pij_idx+7])))
        JuMP.set_start_value(x[5], min(JuMP.upper_bound(x[5]), max(JuMP.lower_bound(x[5]), -(u[pij_idx]^2+u[pij_idx+1]^2))))
        JuMP.set_start_value(x[6], min(JuMP.upper_bound(x[6]), max(JuMP.lower_bound(x[6]), -(u[pij_idx+2]^2+u[pij_idx+3]^2))))

        fr_idx = bus_start + 2*(brBusIdx[2*(I-1)+1] - 1)
        to_idx = bus_start + 2*(brBusIdx[2*I] - 1)

        param[1,I] = l[pij_idx]
        param[2,I] = l[pij_idx+1]
        param[3,I] = l[pij_idx+2]
        param[4,I] = l[pij_idx+3]
        param[5,I] = l[pij_idx+4]
        param[6,I] = l[pij_idx+5]
        param[7,I] = l[pij_idx+6]
        param[8,I] = l[pij_idx+7]
        param[9,I] = rho[pij_idx]
        param[10,I] = rho[pij_idx+1]
        param[11,I] = rho[pij_idx+2]
        param[12,I] = rho[pij_idx+3]
        param[13,I] = rho[pij_idx+4]
        param[14,I] = rho[pij_idx+5]
        param[15,I] = rho[pij_idx+6]
        param[16,I] = rho[pij_idx+7]
        param[17,I] = xbar[xbar_pij_idx] - z[pij_idx]
        param[18,I] = xbar[xbar_pij_idx+1] - z[pij_idx+1]
        param[19,I] = xbar[xbar_pij_idx+2] - z[pij_idx+2]
        param[20,I] = xbar[xbar_pij_idx+3] - z[pij_idx+3]
        param[21,I] = xbar[fr_idx] - z[pij_idx+4]
        param[22,I] = xbar[to_idx] - z[pij_idx+5]
        param[23,I] = xbar[fr_idx+1] - z[pij_idx+6]
        param[24,I] = xbar[to_idx+1] - z[pij_idx+7]

        #=
        JuMP.@variable(jm, pij)
        JuMP.@variable(jm, qij)
        JuMP.@variable(jm, pji)
        JuMP.@variable(jm, qji)
        JuMP.@NLconstraint(jm, pij == YffR*(x[1])^2 + YftR*x[1]*x[2]*cos(x[3]-x[4]) + YftI*x[1]*x[2]*sin(x[3]-x[4]))
        JuMP.@NLconstraint(jm, qij == -YffI*(x[1])^2 - YftI*x[1]*x[2]*cos(x[3]-x[4]) + YftR*x[1]*x[2]*sin(x[3]-x[4]))
        JuMP.@NLconstraint(jm, pji == YttR*(x[2])^2 + YtfR*x[1]*x[2]*cos(x[3]-x[4]) - YtfI*x[1]*x[2]*sin(x[3]-x[4]))
        JuMP.@NLconstraint(jm, qji == -YttI*(x[2])^2 - YtfI*x[1]*x[2]*cos(x[3]-x[4]) - YtfR*x[1]*x[2]*sin(x[3]-x[4]))
        =#

        JuMP.@NLexpression(jm, pij, YffR*(x[1])^2 + YftR*x[1]*x[2]*cos(x[3]-x[4]) + YftI*x[1]*x[2]*sin(x[3]-x[4]))
        JuMP.@NLexpression(jm, qij, -YffI*(x[1])^2 - YftI*x[1]*x[2]*cos(x[3]-x[4]) + YftR*x[1]*x[2]*sin(x[3]-x[4]))
        JuMP.@NLexpression(jm, pji, YttR*(x[2])^2 + YtfR*x[1]*x[2]*cos(x[3]-x[4]) - YtfI*x[1]*x[2]*sin(x[3]-x[4]))
        JuMP.@NLexpression(jm, qji, -YttI*(x[2])^2 - YtfI*x[1]*x[2]*cos(x[3]-x[4]) - YtfR*x[1]*x[2]*sin(x[3]-x[4]))

        JuMP.@NLobjective(jm, Min,
            param[1,I]*pij + param[2,I]*qij +
            param[3,I]*pji + param[4,I]*qji +
            param[5,I]*x[1]^2 + param[6,I]*x[2]^2 +
            param[7,I]*x[3] + param[8,I]*x[4] +
            0.5*(param[9,I]*(pij - param[17,I])^2) + 0.5*(param[10,I]*(qij - param[18,I])^2) +
            0.5*(param[11,I]*(pji - param[19,I])^2) + 0.5*(param[12,I]*(qji - param[20,I])^2) +
            0.5*(param[13,I]*(x[1]^2 - param[21,I])^2) + 0.5*(param[14,I]*(x[2]^2 - param[22,I])^2) +
            0.5*(param[15,I]*(x[3] - param[23,I])^2) + 0.5*(param[16,I]*(x[4] - param[24,I])^2) +
            param[25,I]*(pij^2 + qij^2 + x[5]) + param[26,I]*(pji^2 + qji^2 + x[6]) +
            0.5*(param[27,I]*(pij^2 + qij^2 + x[5])^2) + 0.5*(param[27,I]*(pji^2 + qji^2 + x[6])^2)
        )

        #=
        JuMP.@NLconstraint(jm, pij^2 + qij^2 <= param[29,I])
        JuMP.@NLconstraint(jm, pji^2 + qji^2 <= param[29,I])
        JuMP.optimize!(jm)
        x_val .= JuMP.value.(jm[:x])
        =#

        if major_iter == 1
            param[27,I] = 10.0
            mu = 10.0
        else
            mu = param[27,I]
        end

        eta = 1 / mu^0.1
        omega = 1 / mu
        it = 0
        avg_tron_minor = 0
        terminate = false

        while !terminate
            it += 1

            JuMP.@NLobjective(jm, Min,
                param[1,I]*pij                          + param[2,I]*qij +
                param[3,I]*pji                          + param[4,I]*qji +
                param[5,I]*x[1]^2                       + param[6,I]*x[2]^2 +
                param[7,I]*x[3]                         + param[8,I]*x[4] +
                0.5*(param[9,I]*(pij - param[17,I])^2)  + 0.5*(param[10,I]*(qij - param[18,I])^2) +
                0.5*(param[11,I]*(pji - param[19,I])^2) + 0.5*(param[12,I]*(qji - param[20,I])^2) +
                0.5*(param[13,I]*(x[1]^2 - param[21,I])^2)  + 0.5*(param[14,I]*(x[2]^2 - param[22,I])^2) +
                0.5*(param[15,I]*(x[3] - param[23,I])^2)    + 0.5*(param[16,I]*(x[4] - param[24,I])^2) +
                param[25,I]*(pij^2 + qij^2 + x[5])     + param[26,I]*(pji^2 + qji^2 + x[6]) +
                0.5*(param[27,I]*(pij^2 + qij^2 + x[5])^2) + 0.5*(param[27,I]*(pji^2 + qji^2 + x[6])^2)
            )

            # Solve the branch problem.
            JuMP.optimize!(jm)
            x_val .= JuMP.value.(jm[:x])

            # Check the termination condition.
            vi_vj_cos = x_val[1]*x_val[2]*cos(x_val[3] - x_val[4])
            vi_vj_sin = x_val[1]*x_val[2]*sin(x_val[3] - x_val[4])
            pij_val = YffR*x_val[1]^2 + YftR*vi_vj_cos + YftI*vi_vj_sin
            qij_val = -YffI*x_val[1]^2 - YftI*vi_vj_cos + YftR*vi_vj_sin
            pji_val = YttR*x_val[2]^2 + YtfR*vi_vj_cos - YtfI*vi_vj_sin
            qji_val = -YttI*x_val[2]^2 - YtfI*vi_vj_cos - YtfR*vi_vj_sin

            cviol1 = pij_val^2 + qij_val^2 + x_val[5]
            cviol2 = pji_val^2 + qji_val^2 + x_val[6]

            cnorm = max(abs(cviol1), abs(cviol2))
            #=
            @printf("I = %d cnorm = %.6e mu = %.6e pij^2 + qij^2 = %.6e, x[5] = %.6e pji^2 + qji^2 = %.6e x[6] = %.6e\n",
                     I, cnorm, mu, pij^2 + qij^2, x[5], pji^2 + qji^2, x[6])
            =#

            if cnorm <= eta
                if cnorm <= 1e-6
                    terminate = true
                else
                    param[25,I] += mu*cviol1
                    param[26,I] += mu*cviol2

                    eta = eta / mu^0.9
                    omega  = omega / mu
                end
            else
                mu = min(mu_max, mu*10)
                eta = 1 / mu^0.1
                omega = 1 / mu
                param[27,I] = mu
            end

            if it >= max_auglag
                terminate = true
            end
        end

        vi_vj_cos = x_val[1]*x_val[2]*cos(x_val[3] - x_val[4])
        vi_vj_sin = x_val[1]*x_val[2]*sin(x_val[3] - x_val[4])
        u[pij_idx] = YffR*x_val[1]^2 + YftR*vi_vj_cos + YftI*vi_vj_sin
        u[pij_idx+1] = -YffI*x_val[1]^2 - YftI*vi_vj_cos + YftR*vi_vj_sin
        u[pij_idx+2] = YttR*x_val[2]^2 + YtfR*vi_vj_cos - YtfI*vi_vj_sin
        u[pij_idx+3] = -YttI*x_val[2]^2 - YtfI*vi_vj_cos - YtfR*vi_vj_sin
        u[pij_idx+4] = x_val[1]^2
        u[pij_idx+5] = x_val[2]^2
        u[pij_idx+6] = x_val[3]
        u[pij_idx+7] = x_val[4]
        param[27,I] = mu
    end

    return 0, 0
end
=#