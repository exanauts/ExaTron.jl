function polar_kernel(n::Int,
                    pij_start::Int, qij_start::Int,
                    pji_start::Int, qji_start::Int,
                    wi_i_ij_start::Int, wi_j_ji_start::Int,
                    ti_i_ij_start::Int, ti_j_ji_start::Int,
                    u_curr::CuDeviceArray{Float64,1}, v_curr::CuDeviceArray{Float64,1},
                    l_curr::CuDeviceArray{Float64,1}, rho::CuDeviceArray{Float64,1},
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

    @inbounds begin
        YffR = _YffR[I]; YffI = _YffI[I]
        YftR = _YftR[I]; YftI = _YftI[I]
        YttR = _YttR[I]; YttI = _YttI[I]
        YtfR = _YtfR[I]; YtfI = _YtfI[I]

        xl[1] = sqrt(frBound[2*(I-1)+1])
        xu[1] = sqrt(frBound[2*I])
        xl[2] = sqrt(toBound[2*(I-1)+1])
        xu[2] = sqrt(toBound[2*I])
        xl[3] = -2*pi
        xu[3] = 2*pi
        xl[4] = -2*pi
        xu[4] = 2*pi

        x[1] = min(xu[1], max(xl[1], sqrt(u_curr[wi_i_ij_start+I])))
        x[2] = min(xu[2], max(xl[2], sqrt(u_curr[wi_j_ji_start+I])))
        x[3] = min(xu[3], max(xl[3], u_curr[ti_i_ij_start+I]))
        x[4] = min(xu[4], max(xl[4], u_curr[ti_j_ji_start+I]))

        param[1,I] = l_curr[pij_start+I]
        param[2,I] = l_curr[qij_start+I]
        param[3,I] = l_curr[pji_start+I]
        param[4,I] = l_curr[qji_start+I]
        param[5,I] = l_curr[wi_i_ij_start+I]
        param[6,I] = l_curr[wi_j_ji_start+I]
        param[7,I] = l_curr[ti_i_ij_start+I]
        param[8,I] = l_curr[ti_j_ji_start+I]
        param[9,I] = rho[pij_start+I]
        param[10,I] = rho[qij_start+I]
        param[11,I] = rho[pji_start+I]
        param[12,I] = rho[qji_start+I]
        param[13,I] = rho[wi_i_ij_start+I]
        param[14,I] = rho[wi_j_ji_start+I]
        param[15,I] = rho[ti_i_ij_start+I]
        param[16,I] = rho[ti_j_ji_start+I]
        param[17,I] = v_curr[pij_start+I]
        param[18,I] = v_curr[qij_start+I]
        param[19,I] = v_curr[pji_start+I]
        param[20,I] = v_curr[qji_start+I]
        param[21,I] = v_curr[wi_i_ij_start+I]
        param[22,I] = v_curr[wi_j_ji_start+I]
        param[23,I] = v_curr[ti_i_ij_start+I]
        param[24,I] = v_curr[ti_j_ji_start+I]

        CUDA.sync_threads()

        status, minor_iter = tron_kernel(n, 500, 200, 1e-6, true, x, xl, xu,
                                         param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI)

        vi_vj_cos = x[1]*x[2]*cos(x[3] - x[4])
        vi_vj_sin = x[1]*x[2]*sin(x[3] - x[4])

        u_curr[pij_start+I] = YffR*x[1]^2 + YftR*vi_vj_cos + YftI*vi_vj_sin
        u_curr[qij_start+I] = -YffI*x[1]^2 - YftI*vi_vj_cos + YftR*vi_vj_sin
        u_curr[pji_start+I] = YttR*x[2]^2 + YtfR*vi_vj_cos - YtfI*vi_vj_sin
        u_curr[qji_start+I] = -YttI*x[2]^2 - YtfI*vi_vj_cos - YtfR*vi_vj_sin
        u_curr[wi_i_ij_start+I] = x[1]^2
        u_curr[wi_j_ji_start+I] = x[2]^2
        u_curr[ti_i_ij_start+I] = x[3]
        u_curr[ti_j_ji_start+I] = x[4]
    end

    return
end

function polar_kernel_cpu(n::Int, nline::Int, major_iter::Int, max_auglag::Int,
                          pij_start::Int, qij_start::Int,
                          pji_start::Int, qji_start::Int,
                          wi_i_ij_start::Int, wi_j_ji_start::Int,
                          ti_i_ij_start::Int, ti_j_ji_start::Int,
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
    avg_auglag_it = 0
    avg_minor_it = 0

    x = zeros(4)
    xl = zeros(4)
    xu = zeros(4)

    xl[3] = -2*pi
    xu[3] = 2*pi
    xl[4] = -2*pi
    xu[4] = 2*pi

    @inbounds for I=1:nline
        xl[1] = sqrt(frBound[2*(I-1)+1])
        xu[1] = sqrt(frBound[2*I])
        xl[2] = sqrt(toBound[2*(I-1)+1])
        xu[2] = sqrt(toBound[2*I])

        x[1] = min(xu[1], max(xl[1], sqrt(u_curr[wi_i_ij_start+I])))
        x[2] = min(xu[2], max(xl[2], sqrt(u_curr[wi_j_ji_start+I])))
        x[3] = min(xu[3], max(xl[3], u_curr[ti_i_ij_start+I]))
        x[4] = min(xu[4], max(xl[4], u_curr[ti_j_ji_start+I]))

        param[1,I] = l_curr[pij_start+I]
        param[2,I] = l_curr[qij_start+I]
        param[3,I] = l_curr[pji_start+I]
        param[4,I] = l_curr[qji_start+I]
        param[5,I] = l_curr[wi_i_ij_start+I]
        param[6,I] = l_curr[wi_j_ji_start+I]
        param[7,I] = l_curr[ti_i_ij_start+I]
        param[8,I] = l_curr[ti_j_ji_start+I]
        param[9,I] = rho[pij_start+I]
        param[10,I] = rho[qij_start+I]
        param[11,I] = rho[pji_start+I]
        param[12,I] = rho[qji_start+I]
        param[13,I] = rho[wi_i_ij_start+I]
        param[14,I] = rho[wi_j_ji_start+I]
        param[15,I] = rho[ti_i_ij_start+I]
        param[16,I] = rho[ti_j_ji_start+I]
        param[17,I] = v_curr[pij_start+I]
        param[18,I] = v_curr[qij_start+I]
        param[19,I] = v_curr[pji_start+I]
        param[20,I] = v_curr[qji_start+I]
        param[21,I] = v_curr[wi_i_ij_start+I]
        param[22,I] = v_curr[wi_j_ji_start+I]
        param[23,I] = v_curr[ti_i_ij_start+I]
        param[24,I] = v_curr[ti_j_ji_start+I]

        function eval_f_cb(x)
            f = eval_f_polar_kernel_cpu(I, x, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI)
            return f
        end

        function eval_g_cb(x, g)
            eval_grad_f_polar_kernel_cpu(I, x, g, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI)
            return
        end

        function eval_h_cb(x, mode, rows, cols, scale, lambda, values)
            eval_h_polar_kernel_cpu(I, x, mode, scale, rows, cols, lambda, values,
                                    param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI)
            return
        end

        nele_hess = 10
        tron = ExaTron.createProblem(4, xl, xu, nele_hess, eval_f_cb, eval_g_cb, eval_h_cb;
                                     :tol => 1e-6, :matrix_type => :Dense, :max_minor => 200,
                                     :frtol => 1e-12)

        tron.x .= x
        status = ExaTron.solveProblem(tron)
        x .= tron.x
        avg_minor_it += tron.minor_iter

        cos_ij = cos(x[3] - x[4])
        sin_ij = sin(x[3] - x[4])
        vi_vj_cos = x[1]*x[2]*cos_ij
        vi_vj_sin = x[1]*x[2]*sin_ij

        u_curr[pij_start+I] = YffR[I]*x[1]^2 + YftR[I]*vi_vj_cos + YftI[I]*vi_vj_sin
        u_curr[qij_start+I] = -YffI[I]*x[1]^2 - YftI[I]*vi_vj_cos + YftR[I]*vi_vj_sin
        u_curr[pji_start+I] = YttR[I]*x[2]^2 + YtfR[I]*vi_vj_cos - YtfI[I]*vi_vj_sin
        u_curr[qji_start+I] = -YttI[I]*x[2]^2 - YtfI[I]*vi_vj_cos - YtfR[I]*vi_vj_sin
        u_curr[wi_i_ij_start+I] = x[1]^2
        u_curr[wi_j_ji_start+I] = x[2]^2
        u_curr[ti_i_ij_start+I] = x[3]
        u_curr[ti_j_ji_start+I] = x[4]
    end

    return 0, avg_minor_it / nline
end
