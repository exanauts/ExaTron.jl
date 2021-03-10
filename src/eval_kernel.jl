function eval_f_kernel(n::Int, x::CuDeviceArray{Float64},
                       param::CuDeviceArray{Float64},
                       YffR::Float64, YffI::Float64,
                       YftR::Float64, YftI::Float64,
                       YttR::Float64, YttI::Float64,
                       YtfR::Float64, YtfI::Float64)
                       #=
                       YffR::CuDeviceArray{Float64}, YffI::CuDeviceArray{Float64},
                       YftR::CuDeviceArray{Float64}, YftI::CuDeviceArray{Float64},
                       YttR::CuDeviceArray{Float64}, YttI::CuDeviceArray{Float64},
                       YtfR::CuDeviceArray{Float64}, YtfI::CuDeviceArray{Float64})
                       =#

    # All threads execute the same code.

    I = blockIdx().x
    f = 0

    @inbounds for i=1:6
        f += param[i,I]*x[i]
    end
    @inbounds for i=1:6
        f += 0.5*(param[6+i,I]*(x[i] - param[12+i,I])^2)
    end

    c1 = (x[1] - (YffR*x[5] + YftR*x[7] + YftI*x[8]))
    c2 = (x[2] - (-YffI*x[5] - YftI*x[7] + YftR*x[8]))
    c3 = (x[3] - (YttR*x[6] + YtfR*x[7] - YtfI*x[8]))
    c4 = (x[4] - (-YttI*x[6] - YtfI*x[7] - YtfR*x[8]))
    c5 = (x[7]^2 + x[8]^2 - x[5]*x[6])

    f += param[19,I]*c1
    f += param[20,I]*c2
    f += param[21,I]*c3
    f += param[22,I]*c4
    f += param[23,I]*c5

    raug = param[24,I]
    f += 0.5*raug*c1^2
    f += 0.5*raug*c2^2
    f += 0.5*raug*c3^2
    f += 0.5*raug*c4^2
    f += 0.5*raug*c5^2

    CUDA.sync_threads()
    return f
end

function eval_grad_f_kernel(n::Int, x::CuDeviceArray{Float64}, g::CuDeviceArray{Float64},
                            param::CuDeviceArray{Float64},
                            YffR::Float64, YffI::Float64,
                            YftR::Float64, YftI::Float64,
                            YttR::Float64, YttI::Float64,
                            YtfR::Float64, YtfI::Float64)

    # All threads execute the same code.
    tx = threadIdx().x
    ty = threadIdx().y
    I = blockIdx().x

    c1 = (x[1] - (YffR*x[5] + YftR*x[7] + YftI*x[8]))
    c2 = (x[2] - (-YffI*x[5] - YftI*x[7] + YftR*x[8]))
    c3 = (x[3] - (YttR*x[6] + YtfR*x[7] - YtfI*x[8]))
    c4 = (x[4] - (-YttI*x[6] - YtfI*x[7] - YtfR*x[8]))
    c5 = (x[7]^2 + x[8]^2 - x[5]*x[6])

    g1 = param[1,I] + param[7,I]*(x[1] - param[13,I])
    g2 = param[2,I] + param[8,I]*(x[2] - param[14,I])
    g3 = param[3,I] + param[9,I]*(x[3] - param[15,I])
    g4 = param[4,I] + param[10,I]*(x[4] - param[16,I])
    g5 = param[5,I] + param[11,I]*(x[5] - param[17,I])
    g6 = param[6,I] + param[12,I]*(x[6] - param[18,I])

    raug = param[24,I]
    g1 += param[19,I] + raug*c1
    g2 += param[20,I] + raug*c2
    g3 += param[21,I] + raug*c3
    g4 += param[22,I] + raug*c4

    g5 += param[19,I]*(-YffR) + param[20,I]*(YffI) + param[23,I]*(-x[6]) +
                raug*(-YffR)*c1 + raug*(YffI)*c2 + raug*(-x[6])*c5
    g6 += param[21,I]*(-YttR) + param[22,I]*(YttI) + param[23,I]*(-x[5]) +
                raug*(-YttR)*c3 + raug*(YttI)*c4 + raug*(-x[5])*c5
    g7 = param[19,I]*(-YftR) + param[20,I]*(YftI) + param[21,I]*(-YtfR) +
                param[22,I]*(YtfI) + param[23,I]*(2*x[7]) +
                raug*(-YftR)*c1 + raug*(YftI)*c2 + raug*(-YtfR)*c3 +
                raug*(YtfI)*c4 + raug*(2*x[7])*c5
    g8 = param[19,I]*(-YftI) + param[20,I]*(-YftR) + param[21,I]*(YtfI) +
                param[22,I]*(YtfR) + param[23,I]*(2*x[8]) +
                raug*(-YftI)*c1 + raug*(-YftR)*c2 + raug*(YtfI)*c3 +
                raug*(YtfR)*c4 + raug*(2*x[8])*c5

    g[1] = g1
    g[2] = g2
    g[3] = g3
    g[4] = g4
    g[5] = g5
    g[6] = g6
    g[7] = g7
    g[8] = g8

    CUDA.sync_threads()
    return
end

function eval_h_kernel(n::Int, x::CuDeviceArray{Float64}, A::CuDeviceArray{Float64},
                       param::CuDeviceArray{Float64},
                       YffR::Float64, YffI::Float64,
                       YftR::Float64, YftI::Float64,
                       YttR::Float64, YttI::Float64,
                       YtfR::Float64, YtfI::Float64)

    # All threads execute the same code.
    tx = threadIdx().x
    ty = threadIdx().y
    I = blockIdx().x

    alrect = param[23,I]
    raug = param[24,I]
    c5 = (x[7]^2 + x[8]^2 - x[5]*x[6])

    # 1st column
    if tx == 1 && ty == 1
        A[1] = param[7,I] + raug
        A[5] = raug*(-YffR)
        A[7] = raug*(-YftR)
        A[8] = raug*(-YftI)

        # 2nd columns
        A[n + 2] = param[8,I] + raug
        A[n + 5] = raug*(YffI)
        A[n + 7] = raug*(YftI)
        A[n + 8] = raug*(-YftR)

        # 3rd column
        A[n*2 + 3] = param[9,I] + raug
        A[n*2 + 6] = raug*(-YttR)
        A[n*2 + 7] = raug*(-YtfR)
        A[n*2 + 8] = raug*(YtfI)

        # 4th column
        A[n*3 + 4] = param[10,I] + raug
        A[n*3 + 6] = raug*(YttI)
        A[n*3 + 7] = raug*(YtfI)
        A[n*3 + 8] = raug*(YtfR)

        # 5th column
        A[n*4 + 5] = param[11,I] + raug*(YffR^2) + raug*(YffI^2) + raug*(x[6]^2)
        A[n*4 + 6] = -(alrect + raug*c5) + raug*(x[5]*x[6])
        A[n*4 + 7] = raug*(YffR*YftR) + raug*(YffI*YftI) + raug*((-x[6])*(2*x[7]))
        A[n*4 + 8] = raug*(YffR*YftI) + raug*(YffI*(-YftR)) + raug*((-x[6])*(2*x[8]))

        # 6th column
        A[n*5 + 6] = param[12,I] + raug*(YttR^2) + raug*(YttI^2) + raug*(x[5]^2)
        A[n*5 + 7] = raug*(YttR*YtfR) + raug*(YttI*YtfI) + raug*((-x[5])*(2*x[7]))
        A[n*5 + 8] = raug*((-YttR)*YtfI) + raug*(YttI*YtfR) + raug*((-x[5])*(2*x[8]))

        # 7th column
        A[n*6 + 7] = (alrect + raug*c5)*2 + raug*(YftR^2) + raug*(YftI^2) +
            raug*(YtfR^2) + raug*(YtfI^2) + raug*((2*x[7])*(2*x[7]))
        A[n*6 + 8] = raug*(YftR*YftI) + raug*(YftI*(-YftR)) + raug*((-YtfR)*YtfI) +
            raug*(YtfI*YtfR) + raug*((2*x[7])*(2*x[8]))

        # 8th column
        A[n*7 + 8] = (alrect + raug*c5)*2 + raug*(YftI^2) + raug*(YftR^2) +
            raug*(YtfI^2) + raug*(YtfR^2) + raug*((2*x[8])*(2*x[8]))
    end

    CUDA.sync_threads()

    if tx > ty
        A[n*(tx-1) + ty] = A[n*(ty-1) + tx]
    end

    CUDA.sync_threads()
    return
end

function eval_f_kernel_cpu(I, x, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI)
    f = 0

    @inbounds begin
        f += sum(param[i,I]*x[i] for i=1:6)
        f += param[25,I]*x[9] + param[26,I]*x[10]
        f += 0.5*sum(param[6+i,I]*(x[i] - param[12+i,I])^2 for i=1:6)
        f += 0.5*(param[27,I]*(x[9] - param[29,I])^2 + param[28,I]*(x[10] - param[30,I])^2)

        c1 = (x[1] - (YffR[I]*x[5] + YftR[I]*x[7] + YftI[I]*x[8]))
        c2 = (x[2] - (-YffI[I]*x[5] - YftI[I]*x[7] + YftR[I]*x[8]))
        c3 = (x[3] - (YttR[I]*x[6] + YtfR[I]*x[7] - YtfI[I]*x[8]))
        c4 = (x[4] - (-YttI[I]*x[6] - YtfI[I]*x[7] - YtfR[I]*x[8]))
        c5 = (x[7]^2 + x[8]^2 - x[5]*x[6])
        c6 = x[9] - x[10] - atan(x[8], x[7])

        f += param[19,I]*c1
        f += param[20,I]*c2
        f += param[21,I]*c3
        f += param[22,I]*c4
        f += param[23,I]*c5
        f += param[31,I]*c6

        raug = param[24,I]
        f += 0.5*raug*c1^2
        f += 0.5*raug*c2^2
        f += 0.5*raug*c3^2
        f += 0.5*raug*c4^2
        f += 0.5*raug*c5^2
        f += 0.5*raug*c6^2
    end

    return f
end

function eval_grad_f_kernel_cpu(I, x, g, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI)
    g .= 0.0

    @inbounds begin
        c1 = (x[1] - (YffR[I]*x[5] + YftR[I]*x[7] + YftI[I]*x[8]))
        c2 = (x[2] - (-YffI[I]*x[5] - YftI[I]*x[7] + YftR[I]*x[8]))
        c3 = (x[3] - (YttR[I]*x[6] + YtfR[I]*x[7] - YtfI[I]*x[8]))
        c4 = (x[4] - (-YttI[I]*x[6] - YtfI[I]*x[7] - YtfR[I]*x[8]))
        c5 = (x[7]^2 + x[8]^2 - x[5]*x[6])
        c6 = x[9] - x[10] - atan(x[8], x[7])

        @inbounds for i=1:6
            g[i] += param[i,I] + param[6+i,I]*(x[i] - param[12+i,I])
        end

        # Gradient from angle differences.
        g[9] += param[25,I] + param[27,I]*(x[9] - param[29,I])
        g[10] += param[26,I] + param[28,I]*(x[10] - param[30,I])

        raug = param[24,I]
        g[1] += param[19,I] + raug*c1
        g[2] += param[20,I] + raug*c2
        g[3] += param[21,I] + raug*c3
        g[4] += param[22,I] + raug*c4
        g[5] += param[19,I]*(-YffR[I]) + param[20,I]*(YffI[I]) + param[23,I]*(-x[6]) +
                raug*(-YffR[I])*c1 + raug*(YffI[I])*c2 + raug*(-x[6])*c5
        g[6] += param[21,I]*(-YttR[I]) + param[22,I]*(YttI[I]) + param[23,I]*(-x[5]) +
                raug*(-YttR[I])*c3 + raug*(YttI[I])*c4 + raug*(-x[5])*c5
        g[7] += param[19,I]*(-YftR[I]) + param[20,I]*(YftI[I]) + param[21,I]*(-YtfR[I]) +
                param[22,I]*(YtfI[I]) + param[23,I]*(2*x[7]) +
                raug*(-YftR[I])*c1 + raug*(YftI[I])*c2 + raug*(-YtfR[I])*c3 +
                raug*(YtfI[I])*c4 + raug*(2*x[7])*c5
        g[8] += param[19,I]*(-YftI[I]) + param[20,I]*(-YftR[I]) + param[21,I]*(YtfI[I]) +
                param[22,I]*(YtfR[I]) + param[23,I]*(2*x[8]) +
                raug*(-YftI[I])*c1 + raug*(-YftR[I])*c2 + raug*(YtfI[I])*c3 +
                raug*(YtfR[I])*c4 + raug*(2*x[8])*c5

        # Gradient from angle differences.
        g[7] += (param[31,I] + raug*c6)*(x[8] / (x[7]^2 + x[8]^2))
        g[8] += (-((param[31,I] + raug*c6)*(x[7] / (x[7]^2 + x[8]^2))))
        g[9] += param[31,I] + raug*c6
        g[10] += (-(param[31,I] + raug*c6))
    end

    return
end

function eval_h_kernel_cpu(I, x, mode, scale, rows, cols, lambda, values,
    param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI)

    @inbounds begin
        # Sparsity pattern of lower-triangular of Hessian.
        #     1   2   3   4   5   6   7   8   9   10
        #    ---------------------------------------
        # 1 | x
        # 2 |     x
        # 3 |         x
        # 4 |             x
        # 5 | x   x           x
        # 6 |         x   x   x   x
        # 7 | x   x   x   x   x   x   x
        # 8 | x   x   x   x   x   x   x   x
        # 9 |                         x   x   x
        # 10|                         x   x   x   x
        #    ---------------------------------------
        if mode == :Structure
            # This doesn't need parallel computation.
            # Move this routine somewhere else.

            nz = 1
            # 1st column
            rows[nz] = 1; cols[nz] = 1; nz += 1
            rows[nz] = 5; cols[nz] = 1; nz += 1
            rows[nz] = 7; cols[nz] = 1; nz += 1
            rows[nz] = 8; cols[nz] = 1; nz += 1

            # 2nd column
            rows[nz] = 2; cols[nz] = 2; nz += 1
            rows[nz] = 5; cols[nz] = 2; nz += 1
            rows[nz] = 7; cols[nz] = 2; nz += 1
            rows[nz] = 8; cols[nz] = 2; nz += 1

            # 3rd column
            rows[nz] = 3; cols[nz] = 3; nz += 1
            rows[nz] = 6; cols[nz] = 3; nz += 1
            rows[nz] = 7; cols[nz] = 3; nz += 1
            rows[nz] = 8; cols[nz] = 3; nz += 1

            # 4th column
            rows[nz] = 4; cols[nz] = 4; nz += 1
            rows[nz] = 6; cols[nz] = 4; nz += 1
            rows[nz] = 7; cols[nz] = 4; nz += 1
            rows[nz] = 8; cols[nz] = 4; nz += 1

            # 5th column
            rows[nz] = 5; cols[nz] = 5; nz += 1
            rows[nz] = 6; cols[nz] = 5; nz += 1
            rows[nz] = 7; cols[nz] = 5; nz += 1
            rows[nz] = 8; cols[nz] = 5; nz += 1

            # 6th column
            rows[nz] = 6; cols[nz] = 6; nz += 1
            rows[nz] = 7; cols[nz] = 6; nz += 1
            rows[nz] = 8; cols[nz] = 6; nz += 1

            # 7th column
            rows[nz] = 7; cols[nz] = 7; nz += 1
            rows[nz] = 8; cols[nz] = 7; nz += 1
            rows[nz] = 9; cols[nz] = 7; nz += 1
            rows[nz] = 10; cols[nz] = 7; nz += 1

            # 8th column
            rows[nz] = 8; cols[nz] = 8; nz += 1
            rows[nz] = 9; cols[nz] = 8; nz += 1
            rows[nz] = 10; cols[nz] = 8; nz += 1

            # 9th column
            rows[nz] = 9; cols[nz] = 9; nz += 1
            rows[nz] = 10; cols[nz] = 9; nz += 1

            # 10th column
            rows[nz] = 10; cols[nz] = 10; nz += 1
        else
            nz = 1
            alrect = param[23,I]
            altheta = param[31,I]
            raug = param[24,I]
            c5 = (x[7]^2 + x[8]^2 - x[5]*x[6])
            c6 = x[9] - x[10] - atan(x[8], x[7])

            # 1st column
            # (1,1)
            values[nz] = param[7,I] + raug
            nz += 1
            # (5,1)
            values[nz] = raug*(-YffR[I])
            nz += 1
            # (7,1)
            values[nz] = raug*(-YftR[I])
            nz += 1
            # (8,1)
            values[nz] = raug*(-YftI[I])
            nz += 1

            # 2nd columns
            # (2,2)
            values[nz] = param[8,I] + raug
            nz += 1
            # (5,2)
            values[nz] = raug*(YffI[I])
            nz += 1
            # (7,2)
            values[nz] = raug*(YftI[I])
            nz += 1
            # (8,2)
            values[nz] = raug*(-YftR[I])
            nz += 1

            # 3rd column
            # (3,3)
            values[nz] = param[9,I] + raug
            nz += 1
            # (6,3)
            values[nz] = raug*(-YttR[I])
            nz += 1
            # (7,3)
            values[nz] = raug*(-YtfR[I])
            nz += 1
            # (8,3)
            values[nz] = raug*(YtfI[I])
            nz += 1

            # 4th column
            # (4,4)
            values[nz] = param[10,I] + raug
            nz += 1
            # (6,4)
            values[nz] = raug*(YttI[I])
            nz += 1
            # (7,4)
            values[nz] = raug*(YtfI[I])
            nz += 1
            # (8,4)
            values[nz] = raug*(YtfR[I])
            nz += 1

            # 5th column
            # (5,5)
            values[nz] = param[11,I] + raug*(YffR[I]^2) + raug*(YffI[I]^2) + raug*(x[6]^2)
            nz += 1
            # (6,5)
            values[nz] = -(alrect + raug*c5) + raug*(x[5]*x[6])
            nz += 1
            # (7,5)
            values[nz] = raug*(YffR[I]*YftR[I]) + raug*(YffI[I]*YftI[I]) + raug*((-x[6])*(2*x[7]))
            nz += 1
            # (8,5)
            values[nz] = raug*(YffR[I]*YftI[I]) + raug*(YffI[I]*(-YftR[I])) + raug*((-x[6])*(2*x[8]))
            nz += 1

            # 6th column
            # (6,6)
            values[nz] = param[12,I] + raug*(YttR[I]^2) + raug*(YttI[I]^2) + raug*(x[5]^2)
            nz += 1
            # (7,6)
            values[nz] = raug*(YttR[I]*YtfR[I]) + raug*(YttI[I]*YtfI[I]) + raug*((-x[5])*(2*x[7]))
            nz += 1
            # (8,6)
            values[nz] = raug*((-YttR[I])*YtfI[I]) + raug*(YttI[I]*YtfR[I]) + raug*((-x[5])*(2*x[8]))
            nz += 1

            # 7th column
            # (7,7)
            values[nz] = (alrect + raug*c5)*2 + raug*(YftR[I]^2) + raug*(YftI[I]^2) +
                raug*(YtfR[I]^2) + raug*(YtfI[I]^2) + raug*((2*x[7])*(2*x[7]))
            values[nz] += (altheta + raug*c6)*((-2*x[7]*x[8]) / (x[7]^2 + x[8]^2)^2) # (l6 + rho*c6)*nabla^2_x7x7 c6
            values[nz] += raug*(x[8] / (x[7]^2 + x[8]^2))^2 # rho*(nabla_x7 c6)^2
            nz += 1
            # (8,7)
            values[nz] = raug*(YftR[I]*YftI[I]) + raug*(YftI[I]*(-YftR[I])) + raug*((-YtfR[I])*YtfI[I]) +
                raug*(YtfI[I]*YtfR[I]) + raug*((2*x[7])*(2*x[8]))
            values[nz] += (altheta + raug*c6)*((x[7]^2 - x[8]^2)/(x[7]^2 + x[8]^2)^2) # (l6 + rho*c6)*nabla^2_x8x7 c6
            values[nz] += raug*((x[8]/(x[7]^2 + x[8]^2))*(-x[7]/(x[7]^2 + x[8]^2))) # (rho*(nabla_x7 c6)*(nabla_x8 c6))
            nz += 1
            # (9,7)
            values[nz] = raug*(x[8]/(x[7]^2 + x[8]^2))
            nz += 1
            # (10,7)
            values[nz] = -raug*(x[8]/(x[7]^2 + x[8]^2))
            nz += 1

            # 8th column
            # (8,8)
            values[nz] = (alrect + raug*c5)*2 + raug*(YftI[I]^2) + raug*(YftR[I]^2) +
                raug*(YtfI[I]^2) + raug*(YtfR[I]^2) + raug*((2*x[8])*(2*x[8]))
            values[nz] += (altheta + raug*c6)*((2*x[7]*x[8]) / (x[7]^2 + x[8]^2)^2) # (l6 + rho*c6)*nabla^2_x8x8 c6
            values[nz] += raug*(-x[7]/(x[7]^2 + x[8]^2))^2
            nz += 1
            # (9,8)
            values[nz] = raug*(-x[7]/(x[7]^2 + x[8]^2))
            nz += 1
            # (10,8)
            values[nz] = -raug*(-x[7]/(x[7]^2 + x[8]^2))
            nz += 1

            # 9th column
            # (9,9)
            values[nz] = param[27,I] + raug
            nz += 1
            # (10,9)
            values[nz] = -raug
            nz += 1

            # 10th column
            # (10,10)
            values[nz] = param[28,I] + raug
            nz += 1
        end
    end

    return
end
