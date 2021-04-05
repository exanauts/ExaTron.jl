@inline function eval_f_kernel(n::Int, scale::Float64, x::CuDeviceArray{Float64,1},
                       param::CuDeviceArray{Float64,2},
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
    f = 0.0

    @inbounds for i=1:6
        f += param[i,I]*x[i]
    end
    @inbounds f += param[25,I]*x[9] + param[26,I]*x[10]
    @inbounds for i=1:6
        f += 0.5*(param[6+i,I]*(x[i] - param[12+i,I])^2)
    end
    @inbounds f += 0.5*(param[27,I]*(x[9] - param[29,I])^2 + param[28,I]*(x[10] - param[30,I])^2)

    @inbounds begin
        c1 = (x[1] - (YffR*x[5] + YftR*x[7] + YftI*x[8]))
        c2 = (x[2] - (-YffI*x[5] - YftI*x[7] + YftR*x[8]))
        c3 = (x[3] - (YttR*x[6] + YtfR*x[7] - YtfI*x[8]))
        c4 = (x[4] - (-YttI*x[6] - YtfI*x[7] - YtfR*x[8]))
        c5 = (x[7]^2 + x[8]^2 - x[5]*x[6])
        c6 = (x[9] - x[10] - atan(x[8], x[7]))

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

    f *= scale
    CUDA.sync_threads()
    return f
end

@inline function eval_grad_f_kernel(n::Int, scale::Float64,
                            x::CuDeviceArray{Float64,1}, g::CuDeviceArray{Float64,1},
                            param::CuDeviceArray{Float64,2},
                            YffR::Float64, YffI::Float64,
                            YftR::Float64, YftI::Float64,
                            YttR::Float64, YttI::Float64,
                            YtfR::Float64, YtfI::Float64)

    # All threads execute the same code.
    tx = threadIdx().x
    ty = threadIdx().y
    I = blockIdx().x

    @inbounds begin
        c1 = (x[1] - (YffR*x[5] + YftR*x[7] + YftI*x[8]))
        c2 = (x[2] - (-YffI*x[5] - YftI*x[7] + YftR*x[8]))
        c3 = (x[3] - (YttR*x[6] + YtfR*x[7] - YtfI*x[8]))
        c4 = (x[4] - (-YttI*x[6] - YtfI*x[7] - YtfR*x[8]))
        c5 = (x[7]^2 + x[8]^2 - x[5]*x[6])
        c6 = (x[9] - x[10] - atan(x[8], x[7]))

        g1 = param[1,I] + param[7,I]*(x[1] - param[13,I])
        g2 = param[2,I] + param[8,I]*(x[2] - param[14,I])
        g3 = param[3,I] + param[9,I]*(x[3] - param[15,I])
        g4 = param[4,I] + param[10,I]*(x[4] - param[16,I])
        g5 = param[5,I] + param[11,I]*(x[5] - param[17,I])
        g6 = param[6,I] + param[12,I]*(x[6] - param[18,I])

        g9 = param[25,I] + param[27,I]*(x[9] - param[29,I])
        g10 = param[26,I] + param[28,I]*(x[10] - param[30,I])

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

        g7 += (param[31,I] + raug*c6)*(x[8] / (x[7]^2 + x[8]^2))
        g8 += (-((param[31,I] + raug*c6)*(x[7] / (x[7]^2 + x[8]^2))))
        g9 += param[31,I] + raug*c6
        g10 += (-(param[31,I] + raug*c6))

        if tx == 1 && ty == 1
            g[1] = scale*g1
            g[2] = scale*g2
            g[3] = scale*g3
            g[4] = scale*g4
            g[5] = scale*g5
            g[6] = scale*g6
            g[7] = scale*g7
            g[8] = scale*g8
            g[9] = scale*g9
            g[10] = scale*g10
        end
    end

    CUDA.sync_threads()
    return
end

@inline function eval_h_kernel(n::Int, scale::Float64,
                       x::CuDeviceArray{Float64,1}, A::CuDeviceArray{Float64,2},
                       param::CuDeviceArray{Float64,2},
                       YffR::Float64, YffI::Float64,
                       YftR::Float64, YftI::Float64,
                       YttR::Float64, YttI::Float64,
                       YtfR::Float64, YtfI::Float64)

    # All threads execute the same code.
    tx = threadIdx().x
    ty = threadIdx().y
    I = blockIdx().x

    @inbounds begin
        alrect = param[23,I]
        altheta = param[31,I]
        raug = param[24,I]
        c5 = (x[7]^2 + x[8]^2 - x[5]*x[6])
        c6 = (x[9] - x[10] - atan(x[8], x[7]))

        if tx == 1 && ty == 1
            # 1st column
            A[1,1] = scale*(param[7,I] + raug)
            A[5,1] = scale*(raug*(-YffR))
            A[7,1] = scale*(raug*(-YftR))
            A[8,1] = scale*(raug*(-YftI))

            # 2nd columns
            A[2,2] = scale*(param[8,I] + raug)
            A[5,2] = scale*(raug*(YffI))
            A[7,2] = scale*(raug*(YftI))
            A[8,2] = scale*(raug*(-YftR))

            # 3rd column
            A[3,3] = scale*(param[9,I] + raug)
            A[6,3] = scale*(raug*(-YttR))
            A[7,3] = scale*(raug*(-YtfR))
            A[8,3] = scale*(raug*(YtfI))

            # 4th column
            A[4,4] = scale*(param[10,I] + raug)
            A[6,4] = scale*(raug*(YttI))
            A[7,4] = scale*(raug*(YtfI))
            A[8,4] = scale*(raug*(YtfR))

            # 5th column
            A[5,5] = scale*(param[11,I] + raug*(YffR^2) + raug*(YffI^2) + raug*(x[6]^2))
            A[6,5] = scale*(-(alrect + raug*c5) + raug*(x[5]*x[6]))
            A[7,5] = scale*(raug*(YffR*YftR) + raug*(YffI*YftI) + raug*((-x[6])*(2*x[7])))
            A[8,5] = scale*(raug*(YffR*YftI) + raug*(YffI*(-YftR)) + raug*((-x[6])*(2*x[8])))

            # 6th column
            A[6,6] = scale*(param[12,I] + raug*(YttR^2) + raug*(YttI^2) + raug*(x[5]^2))
            A[7,6] = scale*(raug*(YttR*YtfR) + raug*(YttI*YtfI) + raug*((-x[5])*(2*x[7])))
            A[8,6] = scale*(raug*((-YttR)*YtfI) + raug*(YttI*YtfR) + raug*((-x[5])*(2*x[8])))

            # 7th column
            A[7,7] = scale*((alrect + raug*c5)*2 + raug*(YftR^2) + raug*(YftI^2) +
                raug*(YtfR^2) + raug*(YtfI^2) + raug*((2*x[7])*(2*x[7])) +
                (altheta + raug*c6)*((-2*x[7]*x[8]) / (x[7]^2 + x[8]^2)^2) +
                raug*(x[8] / (x[7]^2 + x[8]^2))^2)
            A[8,7] = scale*(raug*(YftR*YftI) + raug*(YftI*(-YftR)) + raug*((-YtfR)*YtfI) +
                raug*(YtfI*YtfR) + raug*((2*x[7])*(2*x[8])) +
                (altheta + raug*c6)*((x[7]^2 - x[8]^2)/(x[7]^2 + x[8]^2)^2) +
                raug*((x[8]/(x[7]^2 + x[8]^2))*(-x[7]/(x[7]^2 + x[8]^2))))
            A[9,7] = scale*(raug*(x[8]/(x[7]^2 + x[8]^2)))
            A[10,7] = scale*(-raug*(x[8]/(x[7]^2 + x[8]^2)))

            # 8th column
            A[8,8] = scale*((alrect + raug*c5)*2 + raug*(YftI^2) + raug*(YftR^2) +
                raug*(YtfI^2) + raug*(YtfR^2) + raug*((2*x[8])*(2*x[8])) +
                (altheta + raug*c6)*((2*x[7]*x[8]) / (x[7]^2 + x[8]^2)^2) +
                raug*(-x[7]/(x[7]^2 + x[8]^2))^2)
            A[9,8] = scale*(raug*(-x[7]/(x[7]^2 + x[8]^2)))
            A[10,8] = scale*(-raug*(-x[7]/(x[7]^2 + x[8]^2)))

            # 9th column
            A[9,9] = scale*(param[27,I] + raug)
            A[10,9] = scale*(-raug)

            # 10th column
            A[10,10] = scale*(param[28,I] + raug)
        end
    end

    CUDA.sync_threads()

    if tx <= n && ty == 1
        @inbounds for j=1:n
            if tx > j
                A[j,tx] = A[tx,j]
            end
        end
    end

    CUDA.sync_threads()
    return
end

function eval_f_kernel_cpu(I, x, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI)
    f = 0.0

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
        c6 = (x[9] - x[10] - atan(x[8], x[7]))

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
        c6 = (x[9] - x[10] - atan(x[8], x[7]))

        @inbounds for i=1:6
            g[i] += param[i,I] + param[6+i,I]*(x[i] - param[12+i,I])
        end

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
            c6 = (x[9] - x[10] - atan(x[8], x[7]))

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

@inline function eval_f_polar_kernel(n::Int, scale::Float64, x::CuDeviceArray{Float64,1},
                             param::CuDeviceArray{Float64,2},
                             YffR::Float64, YffI::Float64,
                             YftR::Float64, YftI::Float64,
                             YttR::Float64, YttI::Float64,
                             YtfR::Float64, YtfI::Float64)
    I = blockIdx().x
    f = 0.0

    @inbounds begin
        vi_vj_cos = x[1]*x[2]*cos(x[3] - x[4])
        vi_vj_sin = x[1]*x[2]*sin(x[3] - x[4])
        pij = YffR*x[1]^2 + YftR*vi_vj_cos + YftI*vi_vj_sin
        qij = -YffI*x[1]^2 - YftI*vi_vj_cos + YftR*vi_vj_sin
        pji = YttR*x[2]^2 + YtfR*vi_vj_cos - YtfI*vi_vj_sin
        qji = -YttI*x[2]^2 - YtfI*vi_vj_cos - YtfR*vi_vj_sin

        f += param[1,I]*pij
        f += param[2,I]*qij
        f += param[3,I]*pji
        f += param[4,I]*qji
        f += param[5,I]*x[1]^2
        f += param[6,I]*x[2]^2
        f += param[7,I]*x[3]
        f += param[8,I]*x[4]

        f += 0.5*(param[9,I]*(pij - param[17,I])^2)
        f += 0.5*(param[10,I]*(qij - param[18,I])^2)
        f += 0.5*(param[11,I]*(pji - param[19,I])^2)
        f += 0.5*(param[12,I]*(qji - param[20,I])^2)
        f += 0.5*(param[13,I]*(x[1]^2 - param[21,I])^2)
        f += 0.5*(param[14,I]*(x[2]^2 - param[22,I])^2)
        f += 0.5*(param[15,I]*(x[3] - param[23,I])^2)
        f += 0.5*(param[16,I]*(x[4] - param[24,I])^2)
    end

    f *= scale
    CUDA.sync_threads()

    return f
end

@inline function eval_grad_f_polar_kernel(n::Int, scale::Float64, x::CuDeviceArray{Float64,1}, g::CuDeviceArray{Float64,1},
                                  param::CuDeviceArray{Float64,2},
                                  YffR::Float64, YffI::Float64,
                                  YftR::Float64, YftI::Float64,
                                  YttR::Float64, YttI::Float64,
                                  YtfR::Float64, YtfI::Float64)

    tx = threadIdx().x
    ty = threadIdx().y
    I = blockIdx().x

    @inbounds begin
        cos_ij = cos(x[3] - x[4])
        sin_ij = sin(x[3] - x[4])
        vi_vj_cos = x[1]*x[2]*cos_ij
        vi_vj_sin = x[1]*x[2]*sin_ij
        pij = YffR*x[1]^2 + YftR*vi_vj_cos + YftI*vi_vj_sin
        qij = -YffI*x[1]^2 - YftI*vi_vj_cos + YftR*vi_vj_sin
        pji = YttR*x[2]^2 + YtfR*vi_vj_cos - YtfI*vi_vj_sin
        qji = -YttI*x[2]^2 - YtfI*vi_vj_cos - YtfR*vi_vj_sin

        # Derivative with respect to vi.
        dpij_dx = 2*YffR*x[1] + YftR*x[2]*cos_ij + YftI*x[2]*sin_ij
        dqij_dx = -2*YffI*x[1] - YftI*x[2]*cos_ij + YftR*x[2]*sin_ij
        dpji_dx = YtfR*x[2]*cos_ij - YtfI*x[2]*sin_ij
        dqji_dx = -YtfI*x[2]*cos_ij - YtfR*x[2]*sin_ij

        g1 = param[1,I]*(dpij_dx)
        g1 += param[2,I]*(dqij_dx)
        g1 += param[3,I]*(dpji_dx)
        g1 += param[4,I]*(dqji_dx)
        g1 += param[5,I]*(2*x[1])
        g1 += param[9,I]*(pij - param[17,I])*dpij_dx
        g1 += param[10,I]*(qij - param[18,I])*dqij_dx
        g1 += param[11,I]*(pji - param[19,I])*dpji_dx
        g1 += param[12,I]*(qji - param[20,I])*dqji_dx
        g1 += param[13,I]*(x[1]^2 - param[21,I])*(2*x[1])

        # Derivative with respect to vj.
        dpij_dx = YftR*x[1]*cos_ij + YftI*x[1]*sin_ij
        dqij_dx = -YftI*x[1]*cos_ij + YftR*x[1]*sin_ij
        dpji_dx = 2*YttR*x[2] + YtfR*x[1]*cos_ij - YtfI*x[1]*sin_ij
        dqji_dx = -2*YttI*x[2] - YtfI*x[1]*cos_ij - YtfR*x[1]*sin_ij

        g2 = param[1,I]*(dpij_dx)
        g2 += param[2,I]*(dqij_dx)
        g2 += param[3,I]*(dpji_dx)
        g2 += param[4,I]*(dqji_dx)
        g2 += param[6,I]*(2*x[2])
        g2 += param[9,I]*(pij - param[17,I])*dpij_dx
        g2 += param[10,I]*(qij - param[18,I])*dqij_dx
        g2 += param[11,I]*(pji - param[19,I])*dpji_dx
        g2 += param[12,I]*(qji - param[20,I])*dqji_dx
        g2 += param[14,I]*(x[2]^2 - param[22,I])*(2*x[2])

        # Derivative with respect to ti.
        dpij_dx = -YftR*vi_vj_sin + YftI*vi_vj_cos
        dqij_dx = YftI*vi_vj_sin + YftR*vi_vj_cos
        dpji_dx = -YtfR*vi_vj_sin - YtfI*vi_vj_cos
        dqji_dx = YtfI*vi_vj_sin - YtfR*vi_vj_cos

        g3 = param[1,I]*(dpij_dx)
        g3 += param[2,I]*(dqij_dx)
        g3 += param[3,I]*(dpji_dx)
        g3 += param[4,I]*(dqji_dx)
        g3 += param[7,I]
        g3 += param[9,I]*(pij - param[17,I])*dpij_dx
        g3 += param[10,I]*(qij - param[18,I])*dqij_dx
        g3 += param[11,I]*(pji - param[19,I])*dpji_dx
        g3 += param[12,I]*(qji - param[20,I])*dqji_dx
        g3 += param[15,I]*(x[3] - param[23,I])

        # Derivative with respect to tj.

        g4 = param[1,I]*(-dpij_dx)
        g4 += param[2,I]*(-dqij_dx)
        g4 += param[3,I]*(-dpji_dx)
        g4 += param[4,I]*(-dqji_dx)
        g4 += param[8,I]
        g4 += param[9,I]*(pij - param[17,I])*(-dpij_dx)
        g4 += param[10,I]*(qij - param[18,I])*(-dqij_dx)
        g4 += param[11,I]*(pji - param[19,I])*(-dpji_dx)
        g4 += param[12,I]*(qji - param[20,I])*(-dqji_dx)
        g4 += param[16,I]*(x[4] - param[24,I])

        if tx == 1 && ty == 1
            g[1] = scale*g1
            g[2] = scale*g2
            g[3] = scale*g3
            g[4] = scale*g4
        end
    end

    CUDA.sync_threads()

    return
end

@inline function eval_h_polar_kernel(n::Int, scale::Float64, x::CuDeviceArray{Float64,1}, A::CuDeviceArray{Float64,2},
                             param::CuDeviceArray{Float64,2},
                             YffR::Float64, YffI::Float64,
                             YftR::Float64, YftI::Float64,
                             YttR::Float64, YttI::Float64,
                             YtfR::Float64, YtfI::Float64)

    tx = threadIdx().x
    ty = threadIdx().y
    I = blockIdx().x

    @inbounds begin
        cos_ij = cos(x[3] - x[4])
        sin_ij = sin(x[3] - x[4])
        vi_vj_cos = x[1]*x[2]*cos_ij
        vi_vj_sin = x[1]*x[2]*sin_ij
        pij = YffR*x[1]^2 + YftR*vi_vj_cos + YftI*vi_vj_sin
        qij = -YffI*x[1]^2 - YftI*vi_vj_cos + YftR*vi_vj_sin
        pji = YttR*x[2]^2 + YtfR*vi_vj_cos - YtfI*vi_vj_sin
        qji = -YttI*x[2]^2 - YtfI*vi_vj_cos - YtfR*vi_vj_sin

        if tx == 1 && ty == 1
            # d2f_dvidvi

            dpij_dvi = 2*YffR*x[1] + YftR*x[2]*cos_ij + YftI*x[2]*sin_ij
            dqij_dvi = -2*YffI*x[1] - YftI*x[2]*cos_ij + YftR*x[2]*sin_ij
            dpji_dvi = YtfR*x[2]*cos_ij - YtfI*x[2]*sin_ij
            dqji_dvi = -YtfI*x[2]*cos_ij - YtfR*x[2]*sin_ij

            # l_pij * d2pij_dvidvi
            v = param[1,I]*(2*YffR)
            # l_qij * d2qij_dvidvi
            v += param[2,I]*(-2*YffI)
            # l_pji * d2pji_dvidvi = 0
            # l_qji * d2qji_dvidvi = 0
            # l_vi * 2
            v += 2*param[5,I]
            # rho_pij*(dpij_dvi)^2 + rho_pij*(pij - tilde_pij)*d2pij_dvidvi
            v += param[9,I]*(dpij_dvi)^2 + param[9,I]*(pij - param[17,I])*(2*YffR)
            # rho_qij*(dqij_dvi)^2 + rho_qij*(qij - tilde_qij)*d2qij_dvidvi
            v += param[10,I]*(dqij_dvi)^2 + param[10,I]*(qij - param[18,I])*(-2*YffI)
            # rho_pji*(dpji_dvi)^2 + rho_pji*(pji - tilde_pji)*d2pji_dvidvi
            v += param[11,I]*(dpji_dvi)^2
            # rho_qji*(dqji_dvi)^2
            v += param[12,I]*(dqji_dvi)^2
            # (2*rho_vi*vi)*(2*vi) + rho_vi*(vi^2 - tilde_vi)*2
            v += 4*param[13,I]*x[1]^2 + param[13,I]*(x[1]^2 - param[21,I])*2
            A[1,1] = scale*v

            # d2f_dvidvj

            dpij_dvj = YftR*x[1]*cos_ij + YftI*x[1]*sin_ij
            dqij_dvj = -YftI*x[1]*cos_ij + YftR*x[1]*sin_ij
            dpji_dvj = 2*YttR*x[2] + YtfR*x[1]*cos_ij - YtfI*x[1]*sin_ij
            dqji_dvj = -2*YttI*x[2] - YtfI*x[1]*cos_ij - YtfR*x[1]*sin_ij

            d2pij_dvidvj = YftR*cos_ij + YftI*sin_ij
            d2qij_dvidvj = -YftI*cos_ij + YftR*sin_ij
            d2pji_dvidvj = YtfR*cos_ij - YtfI*sin_ij
            d2qji_dvidvj = -YtfI*cos_ij - YtfR*sin_ij

            # l_pij * d2pij_dvidvj
            v = param[1,I]*(d2pij_dvidvj)
            # l_qij * d2qij_dvidvj
            v += param[2,I]*(d2qij_dvidvj)
            # l_pji * d2pji_dvidvj
            v += param[3,I]*(d2pji_dvidvj)
            # l_qji * d2qji_dvidvj
            v += param[4,I]*(d2qji_dvidvj)
            # rho_pij*(dpij_dvj)*dpij_dvi + rho_pij*(pij - tilde_pij)*(d2pij_dvidvj)
            v += param[9,I]*(dpij_dvj)*dpij_dvi + param[9,I]*(pij - param[17,I])*(d2pij_dvidvj)
            # rho_qij*(dqij_dvj)*dqij_dvi + rho_qij*(qij - tilde_qij)*(d2qij_dvidvj)
            v += param[10,I]*(dqij_dvj)*dqij_dvi + param[10,I]*(qij - param[18,I])*(d2qij_dvidvj)
            # rho_pji*(dpji_dvj)*dpji_dvi + rho_pji*(pji - tilde_pji)*(d2pji_dvidvj)
            v += param[11,I]*(dpji_dvj)*dpji_dvi + param[11,I]*(pji - param[19,I])*(d2pji_dvidvj)
            # rho_qji*(dqji_dvj)*dqji_dvi + rho_qji*(qji - tilde_qji)*(d2qji_dvidvj)
            v += param[12,I]*(dqji_dvj)*dqji_dvi + param[12,I]*(qji - param[20,I])*(d2qji_dvidvj)
            A[2,1] = scale*v
            A[1,2] = scale*v

            # d2f_dvidti

            dpij_dti = -YftR*vi_vj_sin + YftI*vi_vj_cos
            dqij_dti = YftI*vi_vj_sin + YftR*vi_vj_cos
            dpji_dti = -YtfR*vi_vj_sin - YtfI*vi_vj_cos
            dqji_dti = YtfI*vi_vj_sin - YtfR*vi_vj_cos

            d2pij_dvidti = -YftR*x[2]*sin_ij + YftI*x[2]*cos_ij
            d2qij_dvidti = YftI*x[2]*sin_ij + YftR*x[2]*cos_ij
            d2pji_dvidti = -YtfR*x[2]*sin_ij - YtfI*x[2]*cos_ij
            d2qji_dvidti = YtfI*x[2]*sin_ij - YtfR*x[2]*cos_ij

            # l_pij * d2pij_dvidti
            v = param[1,I]*(d2pij_dvidti)
            # l_qij * d2qij_dvidti
            v += param[2,I]*(d2qij_dvidti)
            # l_pji * d2pji_dvidti
            v += param[3,I]*(d2pji_dvidti)
            # l_qji * d2qji_dvidti
            v += param[4,I]*(d2qji_dvidti)
            # rho_pij*(dpij_dti)*dpij_dvi + rho_pij*(pij - tilde_pij)*(d2pij_dvidti)
            v += param[9,I]*(dpij_dti)*dpij_dvi + param[9,I]*(pij - param[17,I])*(d2pij_dvidti)
            # rho_qij*(dqij_dti)*dqij_dvi + rho_qij*(qij - tilde_qij)*(d2qij_dvidti)
            v += param[10,I]*(dqij_dti)*dqij_dvi + param[10,I]*(qij - param[18,I])*(d2qij_dvidti)
            # rho_pji*(dpji_dti)*dpji_dvi + rho_pji*(pji - tilde_pji)*(d2pji_dvidti)
            v += param[11,I]*(dpji_dti)*dpji_dvi + param[11,I]*(pji - param[19,I])*(d2pji_dvidti)
            # rho_qji*(dqji_dti)*dqji_dvi + rho_qji*(qji - tilde_qji)*(d2qji_dvidti)
            v += param[12,I]*(dqji_dti)*dqji_dvi + param[12,I]*(qji - param[20,I])*(d2qji_dvidti)
            A[3,1] = scale*v
            A[1,3] = scale*v

            # d2f_dvidtj

            # l_pij * d2pij_dvidtj
            v = param[1,I]*(-d2pij_dvidti)
            # l_qij * d2qij_dvidtj
            v += param[2,I]*(-d2qij_dvidti)
            # l_pji * d2pji_dvidtj
            v += param[3,I]*(-d2pji_dvidti)
            # l_qji * d2qji_dvidtj
            v += param[4,I]*(-d2qji_dvidti)
            # rho_pij*(dpij_dtj)*dpij_dvi + rho_pij*(pij - tilde_pij)*(d2pij_dvidtj)
            v += param[9,I]*(-dpij_dti)*dpij_dvi + param[9,I]*(pij - param[17,I])*(-d2pij_dvidti)
            # rho_qij*(dqij_dtj)*dqij_dvi + rho_qij*(qij - tilde_qij)*(d2qij_dvidtj)
            v += param[10,I]*(-dqij_dti)*dqij_dvi + param[10,I]*(qij - param[18,I])*(-d2qij_dvidti)
            # rho_pji*(dpji_dtj)*dpji_dvi + rho_pji*(pji - tilde_pji)*(d2pji_dvidtj)
            v += param[11,I]*(-dpji_dti)*dpji_dvi + param[11,I]*(pji - param[19,I])*(-d2pji_dvidti)
            # rho_qji*(dqji_dtj)*dqji_dvi + rho_qji*(qji - tilde_qji)*(d2qji_dvidtj)
            v += param[12,I]*(-dqji_dti)*dqji_dvi + param[12,I]*(qji - param[20,I])*(-d2qji_dvidti)
            A[4,1] = scale*v
            A[1,4] = scale*v

            # d2f_dvjdvj

            # l_pij * d2pij_dvjdvj = l_qij * d2qij_dvjdvj = 0 since d2pij_dvjdvj = d2qij_dvjdvj = 0
            # l_pji * d2pji_dvjdvj
            v = param[3,I]*(2*YttR)
            # l_qji * d2qji_dvjdvj
            v += param[4,I]*(-2*YttI)
            # l_vj * 2
            v += param[6,I]*2
            # rho_pij*(dpij_dvj)^2 + rho_pij*(pij - tilde_pij)*(d2pij_dvjdvj)
            v += param[9,I]*(dpij_dvj)^2
            # rho_qij*(dqij_dvj)^2 + rho_qij*(qij - tilde_qij)*(d2qij_dvjdvj)
            v += param[10,I]*(dqij_dvj)^2
            # rho_pji*(dpji_dvj)^2 + rho_pji*(pji - tilde_pji)*(d2pji_dvjdvj)
            v += param[11,I]*(dpji_dvj)^2 + param[11,I]*(pji - param[19,I])*(2*YttR)
            # rho_qji*(dqji_dvj)^2 + rho_qji*(qji - tilde_qji)*(d2qji_dvjdvj)
            v += param[12,I]*(dqji_dvj)^2 + param[12,I]*(qji - param[20,I])*(-2*YttI)
            # (2*rho_vj*vj)*(2*vj) + rho_vj*(vj^2 - tilde_vj)*2
            v += 4*param[14,I]*x[2]^2 + param[14,I]*(x[2]^2 - param[22,I])*2
            A[2,2] = scale*v

            # d2f_dvjdti

            d2pij_dvjdti = (-YftR*x[1]*sin_ij + YftI*x[1]*cos_ij)
            d2qij_dvjdti = (YftI*x[1]*sin_ij + YftR*x[1]*cos_ij)
            d2pji_dvjdti = (-YtfR*x[1]*sin_ij - YtfI*x[1]*cos_ij)
            d2qji_dvjdti = (YtfI*x[1]*sin_ij - YtfR*x[1]*cos_ij)

            # l_pij * d2pij_dvjdti
            v = param[1,I]*(d2pij_dvjdti)
            # l_qij * d2qij_dvjdti
            v += param[2,I]*(d2qij_dvjdti)
            # l_pji * d2pji_dvjdti
            v += param[3,I]*(d2pji_dvjdti)
            # l_qji * d2qji_dvjdti
            v += param[4,I]*(d2qji_dvjdti)
            # rho_pij*(dpij_dti)*dpij_dvj + rho_pij*(pij - tilde_pij)*(d2pij_dvjdti)
            v += param[9,I]*(dpij_dti)*dpij_dvj + param[9,I]*(pij - param[17,I])*d2pij_dvjdti
            # rho_qij*(dqij_dti)*dqij_dvj + rho_qij*(qij - tilde_qij)*(d2qij_dvjdti)
            v += param[10,I]*(dqij_dti)*dqij_dvj + param[10,I]*(qij - param[18,I])*d2qij_dvjdti
            # rho_pji*(dpji_dti)*dpji_dvj + rho_pji*(pji - tilde_pji)*(d2pji_dvjdti)
            v += param[11,I]*(dpji_dti)*dpji_dvj + param[11,I]*(pji - param[19,I])*d2pji_dvjdti
            # rho_qji*(dqji_dti)*dqji_dvj + rho_qji*(qji - tilde_qji)*(d2qji_dvjdti)
            v += param[12,I]*(dqji_dti)*dqji_dvj + param[12,I]*(qji - param[20,I])*d2qji_dvjdti
            A[3,2] = scale*v
            A[2,3] = scale*v

            # d2f_dvjdtj

            # l_pij * d2pij_dvjdtj
            v = param[1,I]*(-d2pij_dvjdti)
            # l_qij * d2qij_dvjdtj
            v += param[2,I]*(-d2qij_dvjdti)
            # l_pji * d2pji_dvjdtj
            v += param[3,I]*(-d2pji_dvjdti)
            # l_qji * d2qji_dvjdtj
            v += param[4,I]*(-d2qji_dvjdti)
            # rho_pij*(dpij_dtj)*dpij_dvj + rho_pij*(pij - tilde_pij)*(d2pij_dvjdtj)
            v += param[9,I]*(-dpij_dti)*dpij_dvj + param[9,I]*(pij - param[17,I])*(-d2pij_dvjdti)
            # rho_qij*(dqij_dtj)*dqij_dvj + rho_qij*(qij - tilde_qij)*(d2qij_dvjdtj)
            v += param[10,I]*(-dqij_dti)*dqij_dvj + param[10,I]*(qij - param[18,I])*(-d2qij_dvjdti)
            # rho_pji*(dpji_dtj)*dpji_dvj + rho_pji*(pji - tilde_pji)*(d2pji_dvjdtj)
            v += param[11,I]*(-dpji_dti)*dpji_dvj + param[11,I]*(pji - param[19,I])*(-d2pji_dvjdti)
            # rho_qji*(dqji_dtj)*dqji_dvj + rho_qji*(qji - tilde_qji)*(d2qji_dvjdtj)
            v += param[12,I]*(-dqji_dti)*dqji_dvj + param[12,I]*(qji - param[20,I])*(-d2qji_dvjdti)
            A[4,2] = scale*v
            A[2,4] = scale*v

            # d2f_dtidti

            d2pij_dtidti = (-YftR*vi_vj_cos - YftI*vi_vj_sin)
            d2qij_dtidti = (YftI*vi_vj_cos - YftR*vi_vj_sin)
            d2pji_dtidti = (-YtfR*vi_vj_cos + YtfI*vi_vj_sin)
            d2qji_dtidti = (YtfI*vi_vj_cos + YtfR*vi_vj_sin)

            # l_pij * d2pij_dtidti
            v = param[1,I]*(d2pij_dtidti)
            # l_qij * d2qij_dtidti
            v += param[2,I]*(d2qij_dtidti)
            # l_pji * d2pji_dtidti
            v += param[3,I]*(d2pji_dtidti)
            # l_qji * d2qji_dtidti
            v += param[4,I]*(d2qji_dtidti)
            # rho_pij*(dpij_dti)^2 + rho_pij*(pij - tilde_pij)*(d2pij_dtidti)
            v += param[9,I]*(dpij_dti)^2 + param[9,I]*(pij - param[17,I])*(d2pij_dtidti)
            # rho_qij*(dqij_dti)^2 + rho_qij*(qij - tilde_qij)*(d2qij_dtidti)
            v += param[10,I]*(dqij_dti)^2 + param[10,I]*(qij - param[18,I])*(d2qij_dtidti)
            # rho_pji*(dpji_dti)^2 + rho_pji*(pji - tilde_pji)*(d2pji_dtidti)
            v += param[11,I]*(dpji_dti)^2 + param[11,I]*(pji - param[19,I])*(d2pji_dtidti)
            # rho_qji*(dqji_dti)^2 + rho_qji*(qji - tilde_qji)*(d2qji_dtidti)
            v += param[12,I]*(dqji_dti)^2 + param[12,I]*(qji - param[20,I])*(d2qji_dtidti)
            # rho_ti
            v += param[15,I]
            A[3,3] = scale*v

            # d2f_dtidtj

            # l_pij * d2pij_dtidtj
            v = param[1,I]*(-d2pij_dtidti)
            # l_qij * d2qij_dtidtj
            v += param[2,I]*(-d2qij_dtidti)
            # l_pji * d2pji_dtidtj
            v += param[3,I]*(-d2pji_dtidti)
            # l_qji * d2qji_dtidtj
            v += param[4,I]*(-d2qji_dtidti)
            # rho_pij*(dpij_dtj)*dpij_dti + rho_pij*(pij - tilde_pij)*(d2pij_dtidtj)
            v += param[9,I]*(-dpij_dti^2) + param[9,I]*(pij - param[17,I])*(-d2pij_dtidti)
            # rho_qij*(dqij_dtj)*dqij_dti + rho_qij*(qij - tilde_qij)*(d2qij_dtidtj)
            v += param[10,I]*(-dqij_dti^2) + param[10,I]*(qij - param[18,I])*(-d2qij_dtidti)
            # rho_pji*(dpji_dtj)*dpji_dti + rho_pji*(pji - tilde_pji)*(d2pji_dtidtj)
            v += param[11,I]*(-dpji_dti^2) + param[11,I]*(pji - param[19,I])*(-d2pji_dtidti)
            # rho_qji*(dqji_dtj)*dqji_dti + rho_qji*(qji - tilde_qji)*(d2qji_dtidtj)
            v += param[12,I]*(-dqji_dti^2) + param[12,I]*(qji - param[20,I])*(-d2qji_dtidti)
            A[4,3] = scale*v
            A[3,4] = scale*v

            # d2f_dtjdtj
            # l_pij * d2pij_dtjdtj
            v = param[1,I]*(d2pij_dtidti)
            # l_qij * d2qij_dtjdtj
            v += param[2,I]*(d2qij_dtidti)
            # l_pji * d2pji_dtjdtj
            v += param[3,I]*(d2pji_dtidti)
            # l_qji * d2qji_dtjdtj
            v += param[4,I]*(d2qji_dtidti)
            # rho_pij*(dpij_dtj)^2 + rho_pij*(pij - tilde_pij)*(d2pij_dtjdtj)
            v += param[9,I]*(dpij_dti^2) + param[9,I]*(pij - param[17,I])*(d2pij_dtidti)
            # rho_qij*(dqij_dtj)^2 + rho_qij*(qij - tilde_qij)*(d2qij_dtjdtj)
            v += param[10,I]*(dqij_dti^2) + param[10,I]*(qij - param[18,I])*(d2qij_dtidti)
            # rho_pji*(dpji_dtj)^2 + rho_pji*(pji - tilde_pji)*(d2pji_dtjdtj)
            v += param[11,I]*(dpji_dti^2) + param[11,I]*(pji - param[19,I])*(d2pji_dtidti)
            # rho_qji*(dqji_dtj)^2 + rho_qji*(qji - tilde_qji)*(d2qji_dtjdtj)
            v += param[12,I]*(dqji_dti^2) + param[12,I]*(qji - param[20,I])*(d2qji_dtidti)
            # rho_tj
            v += param[16,I]
            A[4,4] = scale*v
        end
    end

    CUDA.sync_threads()

    return
end

function eval_f_polar_kernel_cpu(I, x, param, _YffR, _YffI, _YftR, _YftI, _YttR, _YttI, _YtfR, _YtfI)
    f = 0.0

    @inbounds begin
        YffR = _YffR[I]; YffI = _YffI[I]
        YftR = _YftR[I]; YftI = _YftI[I]
        YttR = _YttR[I]; YttI = _YttI[I]
        YtfR = _YtfR[I]; YtfI = _YtfI[I]

        vi_vj_cos = x[1]*x[2]*cos(x[3] - x[4])
        vi_vj_sin = x[1]*x[2]*sin(x[3] - x[4])
        pij = YffR*x[1]^2 + YftR*vi_vj_cos + YftI*vi_vj_sin
        qij = -YffI*x[1]^2 - YftI*vi_vj_cos + YftR*vi_vj_sin
        pji = YttR*x[2]^2 + YtfR*vi_vj_cos - YtfI*vi_vj_sin
        qji = -YttI*x[2]^2 - YtfI*vi_vj_cos - YtfR*vi_vj_sin

        f += param[1,I]*pij
        f += param[2,I]*qij
        f += param[3,I]*pji
        f += param[4,I]*qji
        f += param[5,I]*x[1]^2
        f += param[6,I]*x[2]^2
        f += param[7,I]*x[3]
        f += param[8,I]*x[4]

        f += 0.5*(param[9,I]*(pij - param[17,I])^2)
        f += 0.5*(param[10,I]*(qij - param[18,I])^2)
        f += 0.5*(param[11,I]*(pji - param[19,I])^2)
        f += 0.5*(param[12,I]*(qji - param[20,I])^2)
        f += 0.5*(param[13,I]*(x[1]^2 - param[21,I])^2)
        f += 0.5*(param[14,I]*(x[2]^2 - param[22,I])^2)
        f += 0.5*(param[15,I]*(x[3] - param[23,I])^2)
        f += 0.5*(param[16,I]*(x[4] - param[24,I])^2)
    end

    return f
end

function eval_grad_f_polar_kernel_cpu(I, x, g, param, _YffR, _YffI, _YftR, _YftI, _YttR, _YttI, _YtfR, _YtfI)
    g .= 0.0

    @inbounds begin
        YffR = _YffR[I]; YffI = _YffI[I]
        YftR = _YftR[I]; YftI = _YftI[I]
        YttR = _YttR[I]; YttI = _YttI[I]
        YtfR = _YtfR[I]; YtfI = _YtfI[I]

        cos_ij = cos(x[3] - x[4])
        sin_ij = sin(x[3] - x[4])
        vi_vj_cos = x[1]*x[2]*cos_ij
        vi_vj_sin = x[1]*x[2]*sin_ij
        pij = YffR*x[1]^2 + YftR*vi_vj_cos + YftI*vi_vj_sin
        qij = -YffI*x[1]^2 - YftI*vi_vj_cos + YftR*vi_vj_sin
        pji = YttR*x[2]^2 + YtfR*vi_vj_cos - YtfI*vi_vj_sin
        qji = -YttI*x[2]^2 - YtfI*vi_vj_cos - YtfR*vi_vj_sin

        # Derivative with respect to vi.
        dpij_dx = 2*YffR*x[1] + YftR*x[2]*cos_ij + YftI*x[2]*sin_ij
        dqij_dx = -2*YffI*x[1] - YftI*x[2]*cos_ij + YftR*x[2]*sin_ij
        dpji_dx = YtfR*x[2]*cos_ij - YtfI*x[2]*sin_ij
        dqji_dx = -YtfI*x[2]*cos_ij - YtfR*x[2]*sin_ij

        g[1] = param[1,I]*(dpij_dx)
        g[1] += param[2,I]*(dqij_dx)
        g[1] += param[3,I]*(dpji_dx)
        g[1] += param[4,I]*(dqji_dx)
        g[1] += param[5,I]*(2*x[1])
        g[1] += param[9,I]*(pij - param[17,I])*dpij_dx
        g[1] += param[10,I]*(qij - param[18,I])*dqij_dx
        g[1] += param[11,I]*(pji - param[19,I])*dpji_dx
        g[1] += param[12,I]*(qji - param[20,I])*dqji_dx
        g[1] += param[13,I]*(x[1]^2 - param[21,I])*(2*x[1])

        # Derivative with respect to vj.
        dpij_dx = YftR*x[1]*cos_ij + YftI*x[1]*sin_ij
        dqij_dx = -YftI*x[1]*cos_ij + YftR*x[1]*sin_ij
        dpji_dx = 2*YttR*x[2] + YtfR*x[1]*cos_ij - YtfI*x[1]*sin_ij
        dqji_dx = -2*YttI*x[2] - YtfI*x[1]*cos_ij - YtfR*x[1]*sin_ij

        g[2] = param[1,I]*(dpij_dx)
        g[2] += param[2,I]*(dqij_dx)
        g[2] += param[3,I]*(dpji_dx)
        g[2] += param[4,I]*(dqji_dx)
        g[2] += param[6,I]*(2*x[2])
        g[2] += param[9,I]*(pij - param[17,I])*dpij_dx
        g[2] += param[10,I]*(qij - param[18,I])*dqij_dx
        g[2] += param[11,I]*(pji - param[19,I])*dpji_dx
        g[2] += param[12,I]*(qji - param[20,I])*dqji_dx
        g[2] += param[14,I]*(x[2]^2 - param[22,I])*(2*x[2])

        # Derivative with respect to ti.
        dpij_dx = -YftR*vi_vj_sin + YftI*vi_vj_cos
        dqij_dx = YftI*vi_vj_sin + YftR*vi_vj_cos
        dpji_dx = -YtfR*vi_vj_sin - YtfI*vi_vj_cos
        dqji_dx = YtfI*vi_vj_sin - YtfR*vi_vj_cos

        g[3] = param[1,I]*(dpij_dx)
        g[3] += param[2,I]*(dqij_dx)
        g[3] += param[3,I]*(dpji_dx)
        g[3] += param[4,I]*(dqji_dx)
        g[3] += param[7,I]
        g[3] += param[9,I]*(pij - param[17,I])*dpij_dx
        g[3] += param[10,I]*(qij - param[18,I])*dqij_dx
        g[3] += param[11,I]*(pji - param[19,I])*dpji_dx
        g[3] += param[12,I]*(qji - param[20,I])*dqji_dx
        g[3] += param[15,I]*(x[3] - param[23,I])

        # Derivative with respect to tj.

        g[4] = param[1,I]*(-dpij_dx)
        g[4] += param[2,I]*(-dqij_dx)
        g[4] += param[3,I]*(-dpji_dx)
        g[4] += param[4,I]*(-dqji_dx)
        g[4] += param[8,I]
        g[4] += param[9,I]*(pij - param[17,I])*(-dpij_dx)
        g[4] += param[10,I]*(qij - param[18,I])*(-dqij_dx)
        g[4] += param[11,I]*(pji - param[19,I])*(-dpji_dx)
        g[4] += param[12,I]*(qji - param[20,I])*(-dqji_dx)
        g[4] += param[16,I]*(x[4] - param[24,I])
    end
end

function eval_h_polar_kernel_cpu(I, x, mode, scale, rows, cols, lambda, values,
    param, _YffR, _YffI, _YftR, _YftI, _YttR, _YttI, _YtfR, _YtfI)

    @inbounds begin
        # Sparsity pattern of lower-triangular of Hessian.
        #     1   2   3   4
        #    --------------
        # 1 | x
        # 2 | x   x
        # 3 | x   x   x
        # 4 | x   x   x   x
        #    --------------
        if mode == :Structure
            nz = 1
            for j=1:4
                for i=j:4
                    rows[nz] = i
                    cols[nz] = j
                    nz += 1
                end
            end
        else
            YffR = _YffR[I]; YffI = _YffI[I]
            YftR = _YftR[I]; YftI = _YftI[I]
            YttR = _YttR[I]; YttI = _YttI[I]
            YtfR = _YtfR[I]; YtfI = _YtfI[I]

            cos_ij = cos(x[3] - x[4])
            sin_ij = sin(x[3] - x[4])
            vi_vj_cos = x[1]*x[2]*cos_ij
            vi_vj_sin = x[1]*x[2]*sin_ij
            pij = YffR*x[1]^2 + YftR*vi_vj_cos + YftI*vi_vj_sin
            qij = -YffI*x[1]^2 - YftI*vi_vj_cos + YftR*vi_vj_sin
            pji = YttR*x[2]^2 + YtfR*vi_vj_cos - YtfI*vi_vj_sin
            qji = -YttI*x[2]^2 - YtfI*vi_vj_cos - YtfR*vi_vj_sin

            nz = 1

            # d2f_dvidvi

            dpij_dvi = 2*YffR*x[1] + YftR*x[2]*cos_ij + YftI*x[2]*sin_ij
            dqij_dvi = -2*YffI*x[1] - YftI*x[2]*cos_ij + YftR*x[2]*sin_ij
            dpji_dvi = YtfR*x[2]*cos_ij - YtfI*x[2]*sin_ij
            dqji_dvi = -YtfI*x[2]*cos_ij - YtfR*x[2]*sin_ij

            # l_pij * d2pij_dvidvi
            values[nz] = param[1,I]*(2*YffR)
            # l_qij * d2qij_dvidvi
            values[nz] += param[2,I]*(-2*YffI)
            # l_pji * d2pji_dvidvi = 0
            # l_qji * d2qji_dvidvi = 0
            # l_vi * 2
            values[nz] += 2*param[5,I]
            # rho_pij*(dpij_dvi)^2 + rho_pij*(pij - tilde_pij)*d2pij_dvidvi
            values[nz] += param[9,I]*(dpij_dvi)^2 + param[9,I]*(pij - param[17,I])*(2*YffR)
            # rho_qij*(dqij_dvi)^2 + rho_qij*(qij - tilde_qij)*d2qij_dvidvi
            values[nz] += param[10,I]*(dqij_dvi)^2 + param[10,I]*(qij - param[18,I])*(-2*YffI)
            # rho_pji*(dpji_dvi)^2 + rho_pji*(pji - tilde_pji)*d2pji_dvidvi
            values[nz] += param[11,I]*(dpji_dvi)^2
            # rho_qji*(dqji_dvi)^2
            values[nz] += param[12,I]*(dqji_dvi)^2
            # (2*rho_vi*vi)*(2*vi) + rho_vi*(vi^2 - tilde_vi)*2
            values[nz] += 4*param[13,I]*x[1]^2 + param[13,I]*(x[1]^2 - param[21,I])*2
            nz += 1

            # d2f_dvidvj

            dpij_dvj = YftR*x[1]*cos_ij + YftI*x[1]*sin_ij
            dqij_dvj = -YftI*x[1]*cos_ij + YftR*x[1]*sin_ij
            dpji_dvj = 2*YttR*x[2] + YtfR*x[1]*cos_ij - YtfI*x[1]*sin_ij
            dqji_dvj = -2*YttI*x[2] - YtfI*x[1]*cos_ij - YtfR*x[1]*sin_ij

            d2pij_dvidvj = YftR*cos_ij + YftI*sin_ij
            d2qij_dvidvj = -YftI*cos_ij + YftR*sin_ij
            d2pji_dvidvj = YtfR*cos_ij - YtfI*sin_ij
            d2qji_dvidvj = -YtfI*cos_ij - YtfR*sin_ij

            # l_pij * d2pij_dvidvj
            values[nz] = param[1,I]*(d2pij_dvidvj)
            # l_qij * d2qij_dvidvj
            values[nz] += param[2,I]*(d2qij_dvidvj)
            # l_pji * d2pji_dvidvj
            values[nz] += param[3,I]*(d2pji_dvidvj)
            # l_qji * d2qji_dvidvj
            values[nz] += param[4,I]*(d2qji_dvidvj)
            # rho_pij*(dpij_dvj)*dpij_dvi + rho_pij*(pij - tilde_pij)*(d2pij_dvidvj)
            values[nz] += param[9,I]*(dpij_dvj)*dpij_dvi + param[9,I]*(pij - param[17,I])*(d2pij_dvidvj)
            # rho_qij*(dqij_dvj)*dqij_dvi + rho_qij*(qij - tilde_qij)*(d2qij_dvidvj)
            values[nz] += param[10,I]*(dqij_dvj)*dqij_dvi + param[10,I]*(qij - param[18,I])*(d2qij_dvidvj)
            # rho_pji*(dpji_dvj)*dpji_dvi + rho_pji*(pji - tilde_pji)*(d2pji_dvidvj)
            values[nz] += param[11,I]*(dpji_dvj)*dpji_dvi + param[11,I]*(pji - param[19,I])*(d2pji_dvidvj)
            # rho_qji*(dqji_dvj)*dqji_dvi + rho_qji*(qji - tilde_qji)*(d2qji_dvidvj)
            values[nz] += param[12,I]*(dqji_dvj)*dqji_dvi + param[12,I]*(qji - param[20,I])*(d2qji_dvidvj)
            nz += 1

            # d2f_dvidti

            dpij_dti = -YftR*vi_vj_sin + YftI*vi_vj_cos
            dqij_dti = YftI*vi_vj_sin + YftR*vi_vj_cos
            dpji_dti = -YtfR*vi_vj_sin - YtfI*vi_vj_cos
            dqji_dti = YtfI*vi_vj_sin - YtfR*vi_vj_cos

            d2pij_dvidti = -YftR*x[2]*sin_ij + YftI*x[2]*cos_ij
            d2qij_dvidti = YftI*x[2]*sin_ij + YftR*x[2]*cos_ij
            d2pji_dvidti = -YtfR*x[2]*sin_ij - YtfI*x[2]*cos_ij
            d2qji_dvidti = YtfI*x[2]*sin_ij - YtfR*x[2]*cos_ij

            # l_pij * d2pij_dvidti
            values[nz] = param[1,I]*(d2pij_dvidti)
            # l_qij * d2qij_dvidti
            values[nz] += param[2,I]*(d2qij_dvidti)
            # l_pji * d2pji_dvidti
            values[nz] += param[3,I]*(d2pji_dvidti)
            # l_qji * d2qji_dvidti
            values[nz] += param[4,I]*(d2qji_dvidti)
            # rho_pij*(dpij_dti)*dpij_dvi + rho_pij*(pij - tilde_pij)*(d2pij_dvidti)
            values[nz] += param[9,I]*(dpij_dti)*dpij_dvi + param[9,I]*(pij - param[17,I])*(d2pij_dvidti)
            # rho_qij*(dqij_dti)*dqij_dvi + rho_qij*(qij - tilde_qij)*(d2qij_dvidti)
            values[nz] += param[10,I]*(dqij_dti)*dqij_dvi + param[10,I]*(qij - param[18,I])*(d2qij_dvidti)
            # rho_pji*(dpji_dti)*dpji_dvi + rho_pji*(pji - tilde_pji)*(d2pji_dvidti)
            values[nz] += param[11,I]*(dpji_dti)*dpji_dvi + param[11,I]*(pji - param[19,I])*(d2pji_dvidti)
            # rho_qji*(dqji_dti)*dqji_dvi + rho_qji*(qji - tilde_qji)*(d2qji_dvidti)
            values[nz] += param[12,I]*(dqji_dti)*dqji_dvi + param[12,I]*(qji - param[20,I])*(d2qji_dvidti)
            nz += 1

            # d2f_dvidtj

            # l_pij * d2pij_dvidtj
            values[nz] = param[1,I]*(-d2pij_dvidti)
            # l_qij * d2qij_dvidtj
            values[nz] += param[2,I]*(-d2qij_dvidti)
            # l_pji * d2pji_dvidtj
            values[nz] += param[3,I]*(-d2pji_dvidti)
            # l_qji * d2qji_dvidtj
            values[nz] += param[4,I]*(-d2qji_dvidti)
            # rho_pij*(dpij_dtj)*dpij_dvi + rho_pij*(pij - tilde_pij)*(d2pij_dvidtj)
            values[nz] += param[9,I]*(-dpij_dti)*dpij_dvi + param[9,I]*(pij - param[17,I])*(-d2pij_dvidti)
            # rho_qij*(dqij_dtj)*dqij_dvi + rho_qij*(qij - tilde_qij)*(d2qij_dvidtj)
            values[nz] += param[10,I]*(-dqij_dti)*dqij_dvi + param[10,I]*(qij - param[18,I])*(-d2qij_dvidti)
            # rho_pji*(dpji_dtj)*dpji_dvi + rho_pji*(pji - tilde_pji)*(d2pji_dvidtj)
            values[nz] += param[11,I]*(-dpji_dti)*dpji_dvi + param[11,I]*(pji - param[19,I])*(-d2pji_dvidti)
            # rho_qji*(dqji_dtj)*dqji_dvi + rho_qji*(qji - tilde_qji)*(d2qji_dvidtj)
            values[nz] += param[12,I]*(-dqji_dti)*dqji_dvi + param[12,I]*(qji - param[20,I])*(-d2qji_dvidti)
            nz += 1

            # d2f_dvjdvj

            # l_pij * d2pij_dvjdvj = l_qij * d2qij_dvjdvj = 0 since d2pij_dvjdvj = d2qij_dvjdvj = 0
            # l_pji * d2pji_dvjdvj
            values[nz] = param[3,I]*(2*YttR)
            # l_qji * d2qji_dvjdvj
            values[nz] += param[4,I]*(-2*YttI)
            # l_vj * 2
            values[nz] += param[6,I]*2
            # rho_pij*(dpij_dvj)^2 + rho_pij*(pij - tilde_pij)*(d2pij_dvjdvj)
            values[nz] += param[9,I]*(dpij_dvj)^2
            # rho_qij*(dqij_dvj)^2 + rho_qij*(qij - tilde_qij)*(d2qij_dvjdvj)
            values[nz] += param[10,I]*(dqij_dvj)^2
            # rho_pji*(dpji_dvj)^2 + rho_pji*(pji - tilde_pji)*(d2pji_dvjdvj)
            values[nz] += param[11,I]*(dpji_dvj)^2 + param[11,I]*(pji - param[19,I])*(2*YttR)
            # rho_qji*(dqji_dvj)^2 + rho_qji*(qji - tilde_qji)*(d2qji_dvjdvj)
            values[nz] += param[12,I]*(dqji_dvj)^2 + param[12,I]*(qji - param[20,I])*(-2*YttI)
            # (2*rho_vj*vj)*(2*vj) + rho_vj*(vj^2 - tilde_vj)*2
            values[nz] += 4*param[14,I]*x[2]^2 + param[14,I]*(x[2]^2 - param[22,I])*2
            nz += 1

            # d2f_dvjdti

            d2pij_dvjdti = (-YftR*x[1]*sin_ij + YftI*x[1]*cos_ij)
            d2qij_dvjdti = (YftI*x[1]*sin_ij + YftR*x[1]*cos_ij)
            d2pji_dvjdti = (-YtfR*x[1]*sin_ij - YtfI*x[1]*cos_ij)
            d2qji_dvjdti = (YtfI*x[1]*sin_ij - YtfR*x[1]*cos_ij)

            # l_pij * d2pij_dvjdti
            values[nz] = param[1,I]*(d2pij_dvjdti)
            # l_qij * d2qij_dvjdti
            values[nz] += param[2,I]*(d2qij_dvjdti)
            # l_pji * d2pji_dvjdti
            values[nz] += param[3,I]*(d2pji_dvjdti)
            # l_qji * d2qji_dvjdti
            values[nz] += param[4,I]*(d2qji_dvjdti)
            # rho_pij*(dpij_dti)*dpij_dvj + rho_pij*(pij - tilde_pij)*(d2pij_dvjdti)
            values[nz] += param[9,I]*(dpij_dti)*dpij_dvj + param[9,I]*(pij - param[17,I])*d2pij_dvjdti
            # rho_qij*(dqij_dti)*dqij_dvj + rho_qij*(qij - tilde_qij)*(d2qij_dvjdti)
            values[nz] += param[10,I]*(dqij_dti)*dqij_dvj + param[10,I]*(qij - param[18,I])*d2qij_dvjdti
            # rho_pji*(dpji_dti)*dpji_dvj + rho_pji*(pji - tilde_pji)*(d2pji_dvjdti)
            values[nz] += param[11,I]*(dpji_dti)*dpji_dvj + param[11,I]*(pji - param[19,I])*d2pji_dvjdti
            # rho_qji*(dqji_dti)*dqji_dvj + rho_qji*(qji - tilde_qji)*(d2qji_dvjdti)
            values[nz] += param[12,I]*(dqji_dti)*dqji_dvj + param[12,I]*(qji - param[20,I])*d2qji_dvjdti
            nz += 1

            # d2f_dvjdtj

            # l_pij * d2pij_dvjdtj
            values[nz] = param[1,I]*(-d2pij_dvjdti)
            # l_qij * d2qij_dvjdtj
            values[nz] += param[2,I]*(-d2qij_dvjdti)
            # l_pji * d2pji_dvjdtj
            values[nz] += param[3,I]*(-d2pji_dvjdti)
            # l_qji * d2qji_dvjdtj
            values[nz] += param[4,I]*(-d2qji_dvjdti)
            # rho_pij*(dpij_dtj)*dpij_dvj + rho_pij*(pij - tilde_pij)*(d2pij_dvjdtj)
            values[nz] += param[9,I]*(-dpij_dti)*dpij_dvj + param[9,I]*(pij - param[17,I])*(-d2pij_dvjdti)
            # rho_qij*(dqij_dtj)*dqij_dvj + rho_qij*(qij - tilde_qij)*(d2qij_dvjdtj)
            values[nz] += param[10,I]*(-dqij_dti)*dqij_dvj + param[10,I]*(qij - param[18,I])*(-d2qij_dvjdti)
            # rho_pji*(dpji_dtj)*dpji_dvj + rho_pji*(pji - tilde_pji)*(d2pji_dvjdtj)
            values[nz] += param[11,I]*(-dpji_dti)*dpji_dvj + param[11,I]*(pji - param[19,I])*(-d2pji_dvjdti)
            # rho_qji*(dqji_dtj)*dqji_dvj + rho_qji*(qji - tilde_qji)*(d2qji_dvjdtj)
            values[nz] += param[12,I]*(-dqji_dti)*dqji_dvj + param[12,I]*(qji - param[20,I])*(-d2qji_dvjdti)
            nz += 1

            # d2f_dtidti

            d2pij_dtidti = (-YftR*vi_vj_cos - YftI*vi_vj_sin)
            d2qij_dtidti = (YftI*vi_vj_cos - YftR*vi_vj_sin)
            d2pji_dtidti = (-YtfR*vi_vj_cos + YtfI*vi_vj_sin)
            d2qji_dtidti = (YtfI*vi_vj_cos + YtfR*vi_vj_sin)

            # l_pij * d2pij_dtidti
            values[nz] = param[1,I]*(d2pij_dtidti)
            # l_qij * d2qij_dtidti
            values[nz] += param[2,I]*(d2qij_dtidti)
            # l_pji * d2pji_dtidti
            values[nz] += param[3,I]*(d2pji_dtidti)
            # l_qji * d2qji_dtidti
            values[nz] += param[4,I]*(d2qji_dtidti)
            # rho_pij*(dpij_dti)^2 + rho_pij*(pij - tilde_pij)*(d2pij_dtidti)
            values[nz] += param[9,I]*(dpij_dti)^2 + param[9,I]*(pij - param[17,I])*(d2pij_dtidti)
            # rho_qij*(dqij_dti)^2 + rho_qij*(qij - tilde_qij)*(d2qij_dtidti)
            values[nz] += param[10,I]*(dqij_dti)^2 + param[10,I]*(qij - param[18,I])*(d2qij_dtidti)
            # rho_pji*(dpji_dti)^2 + rho_pji*(pji - tilde_pji)*(d2pji_dtidti)
            values[nz] += param[11,I]*(dpji_dti)^2 + param[11,I]*(pji - param[19,I])*(d2pji_dtidti)
            # rho_qji*(dqji_dti)^2 + rho_qji*(qji - tilde_qji)*(d2qji_dtidti)
            values[nz] += param[12,I]*(dqji_dti)^2 + param[12,I]*(qji - param[20,I])*(d2qji_dtidti)
            # rho_ti
            values[nz] += param[15,I]
            nz += 1

            # d2f_dtidtj

            # l_pij * d2pij_dtidtj
            values[nz] = param[1,I]*(-d2pij_dtidti)
            # l_qij * d2qij_dtidtj
            values[nz] += param[2,I]*(-d2qij_dtidti)
            # l_pji * d2pji_dtidtj
            values[nz] += param[3,I]*(-d2pji_dtidti)
            # l_qji * d2qji_dtidtj
            values[nz] += param[4,I]*(-d2qji_dtidti)
            # rho_pij*(dpij_dtj)*dpij_dti + rho_pij*(pij - tilde_pij)*(d2pij_dtidtj)
            values[nz] += param[9,I]*(-dpij_dti^2) + param[9,I]*(pij - param[17,I])*(-d2pij_dtidti)
            # rho_qij*(dqij_dtj)*dqij_dti + rho_qij*(qij - tilde_qij)*(d2qij_dtidtj)
            values[nz] += param[10,I]*(-dqij_dti^2) + param[10,I]*(qij - param[18,I])*(-d2qij_dtidti)
            # rho_pji*(dpji_dtj)*dpji_dti + rho_pji*(pji - tilde_pji)*(d2pji_dtidtj)
            values[nz] += param[11,I]*(-dpji_dti^2) + param[11,I]*(pji - param[19,I])*(-d2pji_dtidti)
            # rho_qji*(dqji_dtj)*dqji_dti + rho_qji*(qji - tilde_qji)*(d2qji_dtidtj)
            values[nz] += param[12,I]*(-dqji_dti^2) + param[12,I]*(qji - param[20,I])*(-d2qji_dtidti)
            nz += 1

            # d2f_dtjdtj
            # l_pij * d2pij_dtjdtj
            values[nz] = param[1,I]*(d2pij_dtidti)
            # l_qij * d2qij_dtjdtj
            values[nz] += param[2,I]*(d2qij_dtidti)
            # l_pji * d2pji_dtjdtj
            values[nz] += param[3,I]*(d2pji_dtidti)
            # l_qji * d2qji_dtjdtj
            values[nz] += param[4,I]*(d2qji_dtidti)
            # rho_pij*(dpij_dtj)^2 + rho_pij*(pij - tilde_pij)*(d2pij_dtjdtj)
            values[nz] += param[9,I]*(dpij_dti^2) + param[9,I]*(pij - param[17,I])*(d2pij_dtidti)
            # rho_qij*(dqij_dtj)^2 + rho_qij*(qij - tilde_qij)*(d2qij_dtjdtj)
            values[nz] += param[10,I]*(dqij_dti^2) + param[10,I]*(qij - param[18,I])*(d2qij_dtidti)
            # rho_pji*(dpji_dtj)^2 + rho_pji*(pji - tilde_pji)*(d2pji_dtjdtj)
            values[nz] += param[11,I]*(dpji_dti^2) + param[11,I]*(pji - param[19,I])*(d2pji_dtidti)
            # rho_qji*(dqji_dtj)^2 + rho_qji*(qji - tilde_qji)*(d2qji_dtjdtj)
            values[nz] += param[12,I]*(dqji_dti^2) + param[12,I]*(qji - param[20,I])*(d2qji_dtidti)
            # rho_tj
            values[nz] += param[16,I]
        end
    end
end
