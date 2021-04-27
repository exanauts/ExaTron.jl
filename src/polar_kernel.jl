function polar_kernel(n::Int, nlines::Int, line_start::Int, scale::T,
                     u_curr::CuDeviceArray{T,1}, v_curr::CuDeviceArray{T,1},
                     l_curr::CuDeviceArray{T,1}, rho::CuDeviceArray{T,1},
                     shift_lines::Int, param::CuDeviceArray{T,2},
                     _YffR::CuDeviceArray{T,1}, _YffI::CuDeviceArray{T,1},
                     _YftR::CuDeviceArray{T,1}, _YftI::CuDeviceArray{T,1},
                     _YttR::CuDeviceArray{T,1}, _YttI::CuDeviceArray{T,1},
                     _YtfR::CuDeviceArray{T,1}, _YtfI::CuDeviceArray{T,1},
                     frBound::CuDeviceArray{T,1}, toBound::CuDeviceArray{T,1}) where T

    tx = threadIdx().x
    ty = threadIdx().y
    I = blockIdx().x
    id_line = I + shift_lines

    x = @cuDynamicSharedMem(T, n)
    xl = @cuDynamicSharedMem(T, n, n*sizeof(T))
    xu = @cuDynamicSharedMem(T, n, (2*n)*sizeof(T))

    @inbounds begin
        YffR = _YffR[id_line]; YffI = _YffI[id_line]
        YftR = _YftR[id_line]; YftI = _YftI[id_line]
        YttR = _YttR[id_line]; YttI = _YttI[id_line]
        YtfR = _YtfR[id_line]; YtfI = _YtfI[id_line]

        pij_idx = line_start + 8*(I-1)

        xl[1] = CUDA.sqrt(frBound[2*(id_line-1)+1])
        xu[1] = CUDA.sqrt(frBound[2*id_line])
        xl[2] = CUDA.sqrt(toBound[2*(id_line-1)+1])
        xu[2] = CUDA.sqrt(toBound[2*id_line])
        xl[3] = -T(2*pi)
        xu[3] = T(2*pi)
        xl[4] = -T(2*pi)
        xu[4] = T(2*pi)

        x[1] = min(xu[1], max(xl[1], CUDA.sqrt(u_curr[pij_idx+4])))
        x[2] = min(xu[2], max(xl[2], CUDA.sqrt(u_curr[pij_idx+5])))
        x[3] = min(xu[3], max(xl[3], u_curr[pij_idx+6]))
        x[4] = min(xu[4], max(xl[4], u_curr[pij_idx+7]))

        param[1,id_line] = l_curr[pij_idx]
        param[2,id_line] = l_curr[pij_idx+1]
        param[3,id_line] = l_curr[pij_idx+2]
        param[4,id_line] = l_curr[pij_idx+3]
        param[5,id_line] = l_curr[pij_idx+4]
        param[6,id_line] = l_curr[pij_idx+5]
        param[7,id_line] = l_curr[pij_idx+6]
        param[8,id_line] = l_curr[pij_idx+7]
        param[9,id_line] = rho[pij_idx]
        param[10,id_line] = rho[pij_idx+1]
        param[11,id_line] = rho[pij_idx+2]
        param[12,id_line] = rho[pij_idx+3]
        param[13,id_line] = rho[pij_idx+4]
        param[14,id_line] = rho[pij_idx+5]
        param[15,id_line] = rho[pij_idx+6]
        param[16,id_line] = rho[pij_idx+7]
        param[17,id_line] = v_curr[pij_idx]
        param[18,id_line] = v_curr[pij_idx+1]
        param[19,id_line] = v_curr[pij_idx+2]
        param[20,id_line] = v_curr[pij_idx+3]
        param[21,id_line] = v_curr[pij_idx+4]
        param[22,id_line] = v_curr[pij_idx+5]
        param[23,id_line] = v_curr[pij_idx+6]
        param[24,id_line] = v_curr[pij_idx+7]

        CUDA.sync_threads()

        gtol = CUDA.sqrt(eps(T))::T
        tron_kernel(n, shift_lines, 500, 200, gtol, scale, true, x, xl, xu,
                                         param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI)

        vi_vj_cos = x[1]*x[2]*CUDA.cos(x[3] - x[4])
        vi_vj_sin = x[1]*x[2]*CUDA.sin(x[3] - x[4])

        u_curr[pij_idx] = YffR*x[1]^2 + YftR*vi_vj_cos + YftI*vi_vj_sin
        u_curr[pij_idx+1] = -YffI*x[1]^2 - YftI*vi_vj_cos + YftR*vi_vj_sin
        u_curr[pij_idx+2] = YttR*x[2]^2 + YtfR*vi_vj_cos - YtfI*vi_vj_sin
        u_curr[pij_idx+3] = -YttI*x[2]^2 - YtfI*vi_vj_cos - YtfR*vi_vj_sin
        u_curr[pij_idx+4] = x[1]^2
        u_curr[pij_idx+5] = x[2]^2
        u_curr[pij_idx+6] = x[3]
        u_curr[pij_idx+7] = x[4]
    end

    return nothing
end

function polar_kernel_cpu(n::Int, nline::Int, line_start::Int,
                          u_curr::AbstractVector{T}, v_curr::AbstractVector{T},
                          l_curr::AbstractVector{T}, rho::AbstractVector{T},
                          shift::Int,
                          param::Array{T, 2},
                          YffR::Array{T, 1}, YffI::Array{T, 1},
                          YftR::Array{T, 1}, YftI::Array{T, 1},
                          YttR::Array{T, 1}, YttI::Array{T, 1},
                          YtfR::Array{T, 1}, YtfI::Array{T, 1},
                          frBound::Array{T, 1}, toBound::Array{T, 1}) where T
    avg_minor_it = 0

    x = zeros(T, n)
    xl = zeros(T, n)
    xu = zeros(T, n)

    TWOPI = T(2 * pi)

    xl[3] = -TWOPI
    xu[3] = TWOPI
    xl[4] = -TWOPI
    xu[4] = TWOPI

    @inbounds for I=1:nline
        pij_idx = line_start + 8*(I-1)
        id_line = shift + I

        xl[1] = sqrt(frBound[2*(id_line-1)+1])
        xu[1] = sqrt(frBound[2*id_line])
        xl[2] = sqrt(toBound[2*(id_line-1)+1])
        xu[2] = sqrt(toBound[2*id_line])

        x[1] = min(xu[1], max(xl[1], sqrt(u_curr[pij_idx+4])))
        x[2] = min(xu[2], max(xl[2], sqrt(u_curr[pij_idx+5])))
        x[3] = min(xu[3], max(xl[3], u_curr[pij_idx+6]))
        x[4] = min(xu[4], max(xl[4], u_curr[pij_idx+7]))

        param[1,id_line] = l_curr[pij_idx]
        param[2,id_line] = l_curr[pij_idx+1]
        param[3,id_line] = l_curr[pij_idx+2]
        param[4,id_line] = l_curr[pij_idx+3]
        param[5,id_line] = l_curr[pij_idx+4]
        param[6,id_line] = l_curr[pij_idx+5]
        param[7,id_line] = l_curr[pij_idx+6]
        param[8,id_line] = l_curr[pij_idx+7]
        param[9,id_line] = rho[pij_idx]
        param[10,id_line] = rho[pij_idx+1]
        param[11,id_line] = rho[pij_idx+2]
        param[12,id_line] = rho[pij_idx+3]
        param[13,id_line] = rho[pij_idx+4]
        param[14,id_line] = rho[pij_idx+5]
        param[15,id_line] = rho[pij_idx+6]
        param[16,id_line] = rho[pij_idx+7]
        param[17,id_line] = v_curr[pij_idx]
        param[18,id_line] = v_curr[pij_idx+1]
        param[19,id_line] = v_curr[pij_idx+2]
        param[20,id_line] = v_curr[pij_idx+3]
        param[21,id_line] = v_curr[pij_idx+4]
        param[22,id_line] = v_curr[pij_idx+5]
        param[23,id_line] = v_curr[pij_idx+6]
        param[24,id_line] = v_curr[pij_idx+7]

        function eval_f_cb(x)
            f = eval_f_polar_kernel_cpu(id_line, x, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI)
            return f
        end

        function eval_g_cb(x, g)
            eval_grad_f_polar_kernel_cpu(id_line, x, g, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI)
            return
        end

        function eval_h_cb(x, mode, rows, cols, scale, lambda, values)
            eval_h_polar_kernel_cpu(id_line, x, mode, scale, rows, cols, lambda, values,
                                    param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI)
            return
        end

        nele_hess = 10
        atol = sqrt(eps(T))
        frtol = eps(T)^T(2/3)
        tron = ExaTron.createDenseProblem(4, xl, xu, nele_hess, eval_f_cb, eval_g_cb, eval_h_cb;
                                     :tol => atol, :matrix_type => :Dense, :max_minor => 200,
                                     :frtol => frtol)

        tron.x .= x
        status = ExaTron.solveProblem(tron, T)
        x .= tron.x
        avg_minor_it += tron.minor_iter::Int

        cos_ij = cos(x[3] - x[4])
        sin_ij = sin(x[3] - x[4])
        vi_vj_cos = x[1]*x[2]*cos_ij
        vi_vj_sin = x[1]*x[2]*sin_ij

        u_curr[pij_idx] = YffR[id_line]*x[1]^2 + YftR[id_line]*vi_vj_cos + YftI[id_line]*vi_vj_sin
        u_curr[pij_idx+1] = -YffI[id_line]*x[1]^2 - YftI[id_line]*vi_vj_cos + YftR[id_line]*vi_vj_sin
        u_curr[pij_idx+2] = YttR[id_line]*x[2]^2 + YtfR[id_line]*vi_vj_cos - YtfI[id_line]*vi_vj_sin
        u_curr[pij_idx+3] = -YttI[id_line]*x[2]^2 - YtfI[id_line]*vi_vj_cos - YtfR[id_line]*vi_vj_sin
        u_curr[pij_idx+4] = x[1]^2
        u_curr[pij_idx+5] = x[2]^2
        u_curr[pij_idx+6] = x[3]
        u_curr[pij_idx+7] = x[4]
    end

    return 0, avg_minor_it / nline
end
