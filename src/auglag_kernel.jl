function auglag_kernel(n::Int, major_iter::Int, pij_start::Int, qij_start::Int,
                       pji_start::Int, qji_start::Int, wi_i_ij_start::Int, wi_j_ji_start::Int,
                       mu_max::Float64,
                       u_curr::CuDeviceArray{Float64}, v_curr::CuDeviceArray{Float64},
                       l_curr::CuDeviceArray{Float64}, rho::CuDeviceArray{Float64},
                       wRIij::CuDeviceArray{Float64},
                       param::CuDeviceArray{Float64},
                       YffR::CuDeviceArray{Float64}, YffI::CuDeviceArray{Float64},
                       YftR::CuDeviceArray{Float64}, YftI::CuDeviceArray{Float64},
                       YttR::CuDeviceArray{Float64}, YttI::CuDeviceArray{Float64},
                       YtfR::CuDeviceArray{Float64}, YtfI::CuDeviceArray{Float64},
                       frBound::CuDeviceArray{Float64}, toBound::CuDeviceArray{Float64})
    tx = threadIdx().x
    ty = threadIdx().y
    I = blockIdx().x

    x = @cuDynamicSharedMem(Float64, n)
    xl = @cuDynamicSharedMem(Float64, n, n*sizeof(Float64))
    xu = @cuDynamicSharedMem(Float64, n, (2*n)*sizeof(Float64))

    if tx == 1 && ty == 1
        @inbounds for i=1:n
            xl[i] = -Inf
            xu[i] = Inf
        end

        xl[5] = frBound[2*(I-1)+1]
        xu[5] = frBound[2*I]
        xl[6] = toBound[2*(I-1)+1]
        xu[6] = toBound[2*I]

        x[1] = u_curr[pij_start+I]
        x[2] = u_curr[qij_start+I]
        x[3] = u_curr[pji_start+I]
        x[4] = u_curr[qji_start+I]
        x[5] = min(xu[5], max(xl[5], u_curr[wi_i_ij_start+I]))
        x[6] = min(xu[6], max(xu[6], u_curr[wi_j_ji_start+I]))

        if major_iter > 1
            x[7] = wRIij[2*(I-1)+1]
            x[8] = wRIij[2*I]
        else
            x[7] = 0.0
            x[8] = 0.0
        end
    end

    param[1,I] = l_curr[pij_start+I]
    param[2,I] = l_curr[qij_start+I]
    param[3,I] = l_curr[pji_start+I]
    param[4,I] = l_curr[qji_start+I]
    param[5,I] = l_curr[wi_i_ij_start+I]
    param[6,I] = l_curr[wi_j_ji_start+I]
    param[7,I] = rho[pij_start+I]
    param[8,I] = rho[qij_start+I]
    param[9,I] = rho[pji_start+I]
    param[10,I] = rho[qji_start+I]
    param[11,I] = rho[wi_i_ij_start+I]
    param[12,I] = rho[wi_j_ji_start+I]
    param[13,I] = v_curr[pij_start+I]
    param[14,I] = v_curr[qij_start+I]
    param[15,I] = v_curr[pji_start+I]
    param[16,I] = v_curr[qji_start+I]
    param[17,I] = v_curr[wi_i_ij_start+I]
    param[18,I] = v_curr[wi_j_ji_start+I]

    if major_iter == 1
        param[24,I] = 10.0
        mu = 10.0
    else
        mu = param[24,I]
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

        #=
        if tx == 1 && ty == 1
            if it == 6
                @cuprintln("--- [GPU] Iteration 6 starts ---")
            end
        end
        =#

        # Solve the branch problem.
        status, minor_iter = tron_kernel(n, max_feval, max_minor, gtol, x, xl, xu,
                                         param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI)

        # Check the termination condition.
        cviol1 = x[1] - (YffR[I]*x[5] + YftR[I]*x[7] + YftI[I]*x[8])
        cviol2 = x[2] - (-YffI[I]*x[5] - YftI[I]*x[7] + YftR[I]*x[8])
        cviol3 = x[3] - (YttR[I]*x[6] + YtfR[I]*x[7] - YtfI[I]*x[8])
        cviol4 = x[4] - (-YttI[I]*x[6] - YtfI[I]*x[7] - YtfR[I]*x[8])
        cviol5 = x[7]^2 + x[8]^2 - x[5]*x[6]

        cnorm = max(abs(cviol1), abs(cviol2), abs(cviol3), abs(cviol4), abs(cviol5))

        #=
        if tx == 1 && ty == 1
            if it == 6
                @cuprintln("[GPU] Iteration = ", it, " I = ", I, " cnorm = ", cnorm,
                        " mu = ", mu, " eta = ", eta, " status = ", status, " minor_iter = ", minor_iter)
                @inbounds for i=1:n
                    @cuprintln(" x[", i, "] = ", x[i])
                end
                @inbounds for i=19:23
                    @cuprintln(" param[", i, "] = ", param[i,I])
                end
            end
        end
        =#

        if cnorm <= eta
            if cnorm <= 1e-6
                terminate = true
            else
                if tx == 1 && ty == 1
                    param[19,I] += mu*cviol1
                    param[20,I] += mu*cviol2
                    param[21,I] += mu*cviol3
                    param[22,I] += mu*cviol4
                    param[23,I] += mu*cviol5
                end

                eta = eta / CUDA.pow(mu, 0.9)
                omega  = omega / mu
            end
        else
            mu = min(mu_max, mu*10)
            eta = 1 / CUDA.pow(mu, 0.1)
            omega = 1 / mu
            param[24,I] = mu
        end

        if it >= 6
            terminate = true
        end
        CUDA.sync_threads()
    end

    u_curr[pij_start+I] = x[1]
    u_curr[qij_start+I] = x[2]
    u_curr[pji_start+I] = x[3]
    u_curr[qji_start+I] = x[4]
    u_curr[wi_i_ij_start+I] = x[5]
    u_curr[wi_j_ji_start+I] = x[6]
    wRIij[2*(I-1)+1] = x[7]
    wRIij[2*I] = x[8]
    param[24,I] = mu

    CUDA.sync_threads()

    return
end

function auglag_kernel_cpu(n::Int, nline::Int, major_iter::Int, pij_start::Int, qij_start::Int,
                           pji_start::Int, qji_start::Int, wi_i_ij_start::Int, wi_j_ji_start::Int,
                           mu_max::Float64,
                           u_curr::Array{Float64}, v_curr::Array{Float64},
                           l_curr::Array{Float64}, rho::Array{Float64},
                           wRIij::Array{Float64},
                           param::Array{Float64},
                           YffR::Array{Float64}, YffI::Array{Float64},
                           YftR::Array{Float64}, YftI::Array{Float64},
                           YttR::Array{Float64}, YttI::Array{Float64},
                           YtfR::Array{Float64}, YtfI::Array{Float64},
                           frBound::Array{Float64}, toBound::Array{Float64})
    @inbounds for I=1:nline
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

        x = zeros(n)
        xl = zeros(n)
        xu = zeros(n)

        @inbounds for i=1:n
            xl[i] = -Inf
            xu[i] = Inf
        end

        xl[5] = frBound[2*(I-1)+1]
        xu[5] = frBound[2*I]
        xl[6] = toBound[2*(I-1)+1]
        xu[6] = toBound[2*I]

        x[1] = u_curr[pij_start+I]
        x[2] = u_curr[qij_start+I]
        x[3] = u_curr[pji_start+I]
        x[4] = u_curr[qji_start+I]
        x[5] = min(xu[5], max(xl[5], u_curr[wi_i_ij_start+I]))
        x[6] = min(xu[6], max(xu[6], u_curr[wi_j_ji_start+I]))

        if major_iter > 1
            x[7] = wRIij[2*(I-1)+1]
            x[8] = wRIij[2*I]
        else
            x[7] = 0.0
            x[8] = 0.0
        end

        param[1,I] = l_curr[pij_start+I]
        param[2,I] = l_curr[qij_start+I]
        param[3,I] = l_curr[pji_start+I]
        param[4,I] = l_curr[qji_start+I]
        param[5,I] = l_curr[wi_i_ij_start+I]
        param[6,I] = l_curr[wi_j_ji_start+I]
        param[7,I] = rho[pij_start+I]
        param[8,I] = rho[qij_start+I]
        param[9,I] = rho[pji_start+I]
        param[10,I] = rho[qji_start+I]
        param[11,I] = rho[wi_i_ij_start+I]
        param[12,I] = rho[wi_j_ji_start+I]
        param[13,I] = v_curr[pij_start+I]
        param[14,I] = v_curr[qij_start+I]
        param[15,I] = v_curr[pji_start+I]
        param[16,I] = v_curr[qji_start+I]
        param[17,I] = v_curr[wi_i_ij_start+I]
        param[18,I] = v_curr[wi_j_ji_start+I]

        if major_iter == 1
            param[24,I] = 10.0
            mu = 10.0
        else
            mu = param[24,I]
        end

        eta = 1 / mu^0.1
        omega = 1 / mu
        max_feval = 500
        max_minor = 100
        gtol = 1e-6

        nele_hess = 26
        tron = ExaTron.createProblem(n, xl, xu, nele_hess, eval_f_cb, eval_g_cb, eval_h_cb;
                                     :tol => gtol, :matrix_type => :Dense, :max_minor => 200,
                                     :frtol => 1e-10)
        tron.x .= x
        it = 0
        terminate = false

        while !terminate
            it += 1

            #=
            if it == 6
                println("--- [CPU] Iteration 6 starts ---")
            end
            =#

            # Solve the branch problem.
            status = ExaTron.solveProblem(tron)
            x .= tron.x

            # Check the termination condition.
            cviol1 = x[1] - (YffR[I]*x[5] + YftR[I]*x[7] + YftI[I]*x[8])
            cviol2 = x[2] - (-YffI[I]*x[5] - YftI[I]*x[7] + YftR[I]*x[8])
            cviol3 = x[3] - (YttR[I]*x[6] + YtfR[I]*x[7] - YtfI[I]*x[8])
            cviol4 = x[4] - (-YttI[I]*x[6] - YtfI[I]*x[7] - YtfR[I]*x[8])
            cviol5 = x[7]^2 + x[8]^2 - x[5]*x[6]

            cnorm = max(abs(cviol1), abs(cviol2), abs(cviol3), abs(cviol4), abs(cviol5))

            #=
            if it == 6
                println("[CPU] Iteration = ", it, " I = ", I, " cnorm = ", cnorm, " mu = ", mu,
                        " eta = ", eta, " status = ", status, " minor_iter = ", tron.minor_iter)
                @inbounds for i=1:n
                    println(" x[", i, "] = ", x[i])
                end
                @inbounds for i=19:23
                    println(" param[", i, "] = ", param[i,I])
                end
            end
            =#

            if cnorm <= eta
                if cnorm <= 1e-6
                    terminate = true
                else
                    param[19,I] += mu*cviol1
                    param[20,I] += mu*cviol2
                    param[21,I] += mu*cviol3
                    param[22,I] += mu*cviol4
                    param[23,I] += mu*cviol5

                    eta = eta / mu^0.9
                    omega  = omega / mu
                end
            else
                mu = min(mu_max, mu*10)
                eta = 1 / mu^0.1
                omega = 1 / mu
                param[24,I] = mu
            end

            if it >= 6
                terminate = true
            end
        end

        u_curr[pij_start+I] = x[1]
        u_curr[qij_start+I] = x[2]
        u_curr[pji_start+I] = x[3]
        u_curr[qji_start+I] = x[4]
        u_curr[wi_i_ij_start+I] = x[5]
        u_curr[wi_j_ji_start+I] = x[6]
        wRIij[2*(I-1)+1] = x[7]
        wRIij[2*I] = x[8]
        param[24,I] = mu
    end

    return
end
