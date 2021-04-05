function auglag_kernel(n::Int, major_iter::Int, max_auglag::Int,
                       line_start::Int, scale::Float64,
                       mu_max::Float64,
                       u_curr::CuDeviceArray{Float64,1}, v_curr::CuDeviceArray{Float64,1},
                       l_curr::CuDeviceArray{Float64,1}, rho::CuDeviceArray{Float64,1},
                       wRIij::CuDeviceArray{Float64,1},
                       param::CuDeviceArray{Float64,2},
                       _YffR::CuDeviceArray{Float64,1}, _YffI::CuDeviceArray{Float64,1},
                       _YftR::CuDeviceArray{Float64,1}, _YftI::CuDeviceArray{Float64,1},
                       _YttR::CuDeviceArray{Float64,1}, _YttI::CuDeviceArray{Float64,1},
                       _YtfR::CuDeviceArray{Float64,1}, _YtfI::CuDeviceArray{Float64,1},
                       frBound::CuDeviceArray{Float64,1}, toBound::CuDeviceArray{Float64,1})
    tx = threadIdx().x
    ty = threadIdx().y
    I = blockIdx().x

    x = @cuDynamicSharedMem(Float64, n)
    xl = @cuDynamicSharedMem(Float64, n, n*sizeof(Float64))
    xu = @cuDynamicSharedMem(Float64, n, (2*n)*sizeof(Float64))

    pij_idx = line_start + 8*(I-1)

    if tx == 1 && ty == 1
        @inbounds for i=1:n
            xl[i] = -Inf
            xu[i] = Inf
        end

        @inbounds begin
            xl[5] = frBound[2*(I-1)+1]
            xu[5] = frBound[2*I]
            xl[6] = toBound[2*(I-1)+1]
            xu[6] = toBound[2*I]
            xl[9] = -2*pi
            xu[9] = 2*pi
            xl[10] = -2*pi
            xu[10] = 2*pi

            x[1] = u_curr[pij_idx]
            x[2] = u_curr[pij_idx+1]
            x[3] = u_curr[pij_idx+2]
            x[4] = u_curr[pij_idx+3]
            x[5] = min(xu[5], max(xl[5], u_curr[pij_idx+4]))
            x[6] = min(xu[6], max(xl[6], u_curr[pij_idx+5]))
            x[7] = wRIij[2*(I-1)+1]
            x[8] = wRIij[2*I]
            x[9] = min(xu[9], max(xl[9], u_curr[pij_idx+6]))
            x[10] = min(xu[10], max(xl[10], u_curr[pij_idx+7]))
        end
    end

    @inbounds begin
        YffR = _YffR[I]; YffI = _YffI[I]
        YftR = _YftR[I]; YftI = _YftI[I]
        YttR = _YttR[I]; YttI = _YttI[I]
        YtfR = _YtfR[I]; YtfI = _YtfI[I]

        param[1,I] = l_curr[pij_idx]
        param[2,I] = l_curr[pij_idx+1]
        param[3,I] = l_curr[pij_idx+2]
        param[4,I] = l_curr[pij_idx+3]
        param[5,I] = l_curr[pij_idx+4]
        param[6,I] = l_curr[pij_idx+5]
        param[7,I] = rho[pij_idx]
        param[8,I] = rho[pij_idx+1]
        param[9,I] = rho[pij_idx+2]
        param[10,I] = rho[pij_idx+3]
        param[11,I] = rho[pij_idx+4]
        param[12,I] = rho[pij_idx+5]
        param[13,I] = v_curr[pij_idx]
        param[14,I] = v_curr[pij_idx+1]
        param[15,I] = v_curr[pij_idx+2]
        param[16,I] = v_curr[pij_idx+3]
        param[17,I] = v_curr[pij_idx+4]
        param[18,I] = v_curr[pij_idx+5]

        param[25,I] = l_curr[pij_idx+6]
        param[26,I] = l_curr[pij_idx+7]
        param[27,I] = rho[pij_idx+6]
        param[28,I] = rho[pij_idx+7]
        param[29,I] = v_curr[pij_idx+6]
        param[30,I] = v_curr[pij_idx+7]
    end

    if major_iter == 1
        @inbounds param[24,I] = 10.0
        mu = 10.0
    else
        @inbounds mu = param[24,I]
    end

    CUDA.sync_threads()

    eta = 1 / CUDA.pow(mu, 0.1)
    omega = 1 / mu
    max_feval = 500
    max_minor = 200
    gtol = 1e-6

    it = 0
    terminate = false

    while !terminate
        it += 1

        # Solve the branch problem.
        status, minor_iter = tron_kernel(n, max_feval, max_minor, gtol, scale, false, x, xl, xu,
                                         param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI)

        @inbounds begin
            # Check the termination condition.
            cviol1 = x[1] - (YffR*x[5] + YftR*x[7] + YftI*x[8])
            cviol2 = x[2] - (-YffI*x[5] - YftI*x[7] + YftR*x[8])
            cviol3 = x[3] - (YttR*x[6] + YtfR*x[7] - YtfI*x[8])
            cviol4 = x[4] - (-YttI*x[6] - YtfI*x[7] - YtfR*x[8])
            cviol5 = x[7]^2 + x[8]^2 - x[5]*x[6]
            cviol6 = x[9] - x[10] - atan(x[8], x[7])
        end

        cnorm = max(abs(cviol1), abs(cviol2), abs(cviol3), abs(cviol4), abs(cviol5), abs(cviol6))

        if cnorm <= eta
            if cnorm <= 1e-6
                terminate = true
            else
                if tx == 1 && ty == 1
                    @inbounds begin
                        param[19,I] += mu*cviol1
                        param[20,I] += mu*cviol2
                        param[21,I] += mu*cviol3
                        param[22,I] += mu*cviol4
                        param[23,I] += mu*cviol5
                        param[31,I] += mu*cviol6
                    end
                end

                eta = eta / CUDA.pow(mu, 0.9)
                omega  = omega / mu
            end
        else
            mu = min(mu_max, mu*10)
            eta = 1 / CUDA.pow(mu, 0.1)
            omega = 1 / mu
            @inbounds param[24,I] = mu
        end

        if it >= max_auglag
            terminate = true
        end

        CUDA.sync_threads()
    end

    @inbounds begin
        u_curr[pij_idx] = x[1]
        u_curr[pij_idx+1] = x[2]
        u_curr[pij_idx+2] = x[3]
        u_curr[pij_idx+3] = x[4]
        u_curr[pij_idx+4] = x[5]
        u_curr[pij_idx+5] = x[6]
        wRIij[2*(I-1)+1] = x[7]
        wRIij[2*I] = x[8]
        u_curr[pij_idx+6] = x[9]
        u_curr[pij_idx+7] = x[10]
        param[24,I] = mu
    end

    CUDA.sync_threads()

    return
end

function auglag_kernel_cpu(n::Int, nline::Int, major_iter::Int, max_auglag::Int,
                           line_start::Int, mu_max::Float64,
                           u_curr::Array{Float64}, v_curr::Array{Float64},
                           l_curr::Array{Float64}, rho::Array{Float64},
                           wRIij::Array{Float64},
                           param::Array{Float64},
                           YffR::Array{Float64}, YffI::Array{Float64},
                           YftR::Array{Float64}, YftI::Array{Float64},
                           YttR::Array{Float64}, YttI::Array{Float64},
                           YtfR::Array{Float64}, YtfI::Array{Float64},
                           frBound::Array{Float64}, toBound::Array{Float64})
    avg_auglag_it = 0
    avg_minor_it = 0

    x = zeros(n)
    xl = zeros(n)
    xu = zeros(n)

    @inbounds for I=1:nline
        @inbounds for i=1:n
            xl[i] = -Inf
            xu[i] = Inf
        end

        xl[5] = frBound[2*(I-1)+1]
        xu[5] = frBound[2*I]
        xl[6] = toBound[2*(I-1)+1]
        xu[6] = toBound[2*I]
        xl[9] = -2*pi
        xu[9] = 2*pi
        xl[10] = -2*pi
        xu[10] = 2*pi

        pij_idx = line_start + 8*(I-1)

        x[1] = u_curr[pij_idx]
        x[2] = u_curr[pij_idx+1]
        x[3] = u_curr[pij_idx+2]
        x[4] = u_curr[pij_idx+3]
        x[5] = min(xu[5], max(xl[5], u_curr[pij_idx+4]))
        x[6] = min(xu[6], max(xl[6], u_curr[pij_idx+5]))
        x[7] = wRIij[2*(I-1)+1]
        x[8] = wRIij[2*I]
        x[9] = min(xu[9], max(xl[9], u_curr[pij_idx+6]))
        x[10] = min(xu[10], max(xl[10], u_curr[pij_idx+7]))

        param[1,I] = l_curr[pij_idx]
        param[2,I] = l_curr[pij_idx+1]
        param[3,I] = l_curr[pij_idx+2]
        param[4,I] = l_curr[pij_idx+3]
        param[5,I] = l_curr[pij_idx+4]
        param[6,I] = l_curr[pij_idx+5]
        param[7,I] = rho[pij_idx]
        param[8,I] = rho[pij_idx+1]
        param[9,I] = rho[pij_idx+2]
        param[10,I] = rho[pij_idx+3]
        param[11,I] = rho[pij_idx+4]
        param[12,I] = rho[pij_idx+5]
        param[13,I] = v_curr[pij_idx]
        param[14,I] = v_curr[pij_idx+1]
        param[15,I] = v_curr[pij_idx+2]
        param[16,I] = v_curr[pij_idx+3]
        param[17,I] = v_curr[pij_idx+4]
        param[18,I] = v_curr[pij_idx+5]

        if major_iter == 1
            param[24,I] = 10.0
            mu = 10.0
        else
            mu = param[24,I]
        end

        param[25,I] = l_curr[pij_idx+6]
        param[26,I] = l_curr[pij_idx+7]
        param[27,I] = rho[pij_idx+6]
        param[28,I] = rho[pij_idx+7]
        param[29,I] = v_curr[pij_idx+6]
        param[30,I] = v_curr[pij_idx+7]

        function eval_f_cb(x)
            f= eval_f_kernel_cpu(I, x, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI)
            return f
        end

        function eval_g_cb(x, g)
            eval_grad_f_kernel_cpu(I, x, g, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI)
            return
        end

        function eval_h_cb(x, mode, rows, cols, scale, lambda, values)
            eval_h_kernel_cpu(I, x, mode, scale, rows, cols, lambda, values,
                              param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI)
            return
        end

        eta = 1 / mu^0.1
        omega = 1 / mu
        max_feval = 500
        max_minor = 100
        gtol = 1e-6

        nele_hess = 33
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
            x .= tron.x
            avg_tron_minor += tron.minor_iter

            # Check the termination condition.
            cviol1 = x[1] - (YffR[I]*x[5] + YftR[I]*x[7] + YftI[I]*x[8])
            cviol2 = x[2] - (-YffI[I]*x[5] - YftI[I]*x[7] + YftR[I]*x[8])
            cviol3 = x[3] - (YttR[I]*x[6] + YtfR[I]*x[7] - YtfI[I]*x[8])
            cviol4 = x[4] - (-YttI[I]*x[6] - YtfI[I]*x[7] - YtfR[I]*x[8])
            cviol5 = x[7]^2 + x[8]^2 - x[5]*x[6]
            cviol6 = x[9] - x[10] - atan(x[8], x[7])

            cnorm = max(abs(cviol1), abs(cviol2), abs(cviol3), abs(cviol4), abs(cviol5), abs(cviol6))

            if cnorm <= eta
                if cnorm <= 1e-6
                    terminate = true
                else
                    param[19,I] += mu*cviol1
                    param[20,I] += mu*cviol2
                    param[21,I] += mu*cviol3
                    param[22,I] += mu*cviol4
                    param[23,I] += mu*cviol5
                    param[31,I] += mu*cviol6

                    eta = eta / mu^0.9
                    omega  = omega / mu
                end
            else
                mu = min(mu_max, mu*10)
                eta = 1 / mu^0.1
                omega = 1 / mu
                param[24,I] = mu
            end

            if it >= max_auglag
                terminate = true
            end
        end

        avg_auglag_it += it
        avg_minor_it += (avg_tron_minor / it)
        u_curr[pij_idx] = x[1]
        u_curr[pij_idx+1] = x[2]
        u_curr[pij_idx+2] = x[3]
        u_curr[pij_idx+3] = x[4]
        u_curr[pij_idx+4] = x[5]
        u_curr[pij_idx+5] = x[6]
        wRIij[2*(I-1)+1] = x[7]
        wRIij[2*I] = x[8]
        u_curr[pij_idx+6] = x[9]
        u_curr[pij_idx+7] = x[10]
        param[24,I] = mu
    end

    return (avg_auglag_it / nline), (avg_minor_it / nline)
end
