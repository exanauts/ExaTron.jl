function rho_kernel(
    n, k, Kf, Kf_mean, pq_end, eps_rp, eps_rp_min, rt_inc, rt_dec, eta,
    rho_max, rho_min_pq, rho_min_w,
    u_curr::CuDeviceArray{Float64}, u_prev::CuDeviceArray{Float64},
    v_curr::CuDeviceArray{Float64}, v_prev::CuDeviceArray{Float64},
    l_curr::CuDeviceArray{Float64}, l_prev::CuDeviceArray{Float64},
    rho::CuDeviceArray{Float64}, tau::CuDeviceArray{Float64},
    rp::CuDeviceArray{Float64}, rp_old::CuDeviceArray{Float64},
    rp_k0::CuDeviceArray{Float64}
)
    I = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if I <= n
        @inbounds begin
            delta_u = u_curr[I] - u_prev[I]
            delta_v = v_curr[I] - v_prev[I]
            delta_l = l_curr[I] - l_prev[I]
            alpha = abs(delta_l / delta_u)
            beta = abs(delta_l / delta_v)
            rho_v = rho[I]
            rp_v = rp[I]
            rp_old_v = rp_old[I]
            rp_k0_v = rp_k0[I]

            if abs(delta_l) <= eps_rp_min
                tau[I,k] = tau[I,k-1]
            elseif abs(delta_u) <= eps_rp_min && abs(delta_v) > eps_rp_min
                tau[I,k] = beta
            elseif abs(delta_u) > eps_rp_min && abs(delta_v) <= eps_rp_min
                tau[I,k] = alpha
            elseif abs(delta_u) <= eps_rp_min && abs(delta_v) <= eps_rp_min
                tau[I,k] = tau[I,k-1]
            else
                tau[I,k] = sqrt(alpha*beta)
            end

            if (k % Kf) == 0
                mean_tau = 0
                @inbounds for j=1:Kf_mean
                    mean_tau += tau[I,k-j+1]
                end
                mean_tau /= Kf_mean
                if mean_tau >= rt_inc*rho_v
                    if abs(rp_v) >= eps_rp && abs(rp_old_v) >= eps_rp
                        if abs(rp_v) > eta*abs(rp_k0_v) || abs(rp_old_v) > eta*abs(rp_k0_v)
                            rho_v *= rt_inc
                        end
                    end
                    #=
                elseif mean_tau > rt_inc2*rho_v
                    if abs(rp_v) >= eps_rp && abs(rp_old_v) >= eps_rp
                        if abs(rp_v) > eta*abs(rp_k0_v) || abs(rp_old_v) > eta*abs(rp_k0_v)
                            rho_v = rt_inc2*mean_tau
                        end
                    end
                    =#
                elseif mean_tau > rho_v
                    if abs(rp_v) >= eps_rp && abs(rp_old_v) >= eps_rp
                        if abs(rp_v) > eta*abs(rp_k0_v) || abs(rp_old_v) > eta*abs(rp_k0_v)
                            rho_v = mean_tau
                        end
                    end
                elseif mean_tau <= rho_v/rt_dec
                    rho_v /= rt_dec
                    #=
                elseif mean_tau < rho_v/rt_dec2
                    rho_v = mean_tau/rt_dec2
                    =#
                elseif mean_tau < rho_v
                    rho_v = mean_tau
                end
            end

            rho_v = min(rho_max, rho_v)
            if I <= pq_end
                rho_v = max(rho_min_pq, rho_v)
            else
                rho_v = max(rho_min_w, rho_v)
            end
            rho[I] = rho_v
        end
    end

    return
end

function rho_kernel_cpu(
    n, k, Kf, Kf_mean, pq_end, eps_rp, eps_rp_min, rt_inc, rt_dec, eta,
    rho_max, rho_min_pq, rho_min_w,
    u_curr, u_prev, v_curr, v_prev, l_curr, l_prev, rho,
    tau, rp, rp_old, rp_k0
)
    Threads.@threads for I=1:n
        @inbounds begin
            delta_u = u_curr[I] - u_prev[I]
            delta_v = v_curr[I] - v_prev[I]
            delta_l = l_curr[I] - l_prev[I]
            alpha = abs(delta_l / delta_u)
            beta = abs(delta_l / delta_v)
            rho_v = rho[I]
            rp_v = rp[I]
            rp_old_v = rp_old[I]
            rp_k0_v = rp_k0[I]

            if abs(delta_l) <= eps_rp_min
                tau[I,k] = tau[I,k-1]
            elseif abs(delta_u) <= eps_rp_min && abs(delta_v) > eps_rp_min
                tau[I,k] = beta
            elseif abs(delta_u) > eps_rp_min && abs(delta_v) <= eps_rp_min
                tau[I,k] = alpha
            elseif abs(delta_u) <= eps_rp_min && abs(delta_v) <= eps_rp_min
                tau[I,k] = tau[I,k-1]
            else
                tau[I,k] = sqrt(alpha*beta)
            end

            if (k % Kf) == 0
                mean_tau = mean(tau[I,k - Kf_mean+1:k])
                if mean_tau >= rt_inc*rho_v
                    if abs(rp_v) >= eps_rp && abs(rp_old_v) >= eps_rp
                        if abs(rp_v) > eta*abs(rp_k0_v) || abs(rp_old_v) > eta*abs(rp_k0_v)
                            rho_v *= rt_inc
                        end
                    end
                    #=
                elseif mean_tau > rt_inc2*rho_v
                    if abs(rp_v) >= eps_rp && abs(rp_old_v) >= eps_rp
                        if abs(rp_v) > eta*abs(rp_k0_v) || abs(rp_old_v) > eta*abs(rp_k0_v)
                            rho_v = rt_inc2*mean_tau
                        end
                    end
                    =#
                elseif mean_tau > rho_v
                    if abs(rp_v) >= eps_rp && abs(rp_old_v) >= eps_rp
                        if abs(rp_v) > eta*abs(rp_k0_v) || abs(rp_old_v) > eta*abs(rp_k0_v)
                            rho_v = mean_tau
                        end
                    end
                elseif mean_tau <= rho_v/rt_dec
                    rho_v /= rt_dec
                    #=
                elseif mean_tau < rho_v/rt_dec2
                    rho_v = mean_tau/rt_dec2
                    =#
                elseif mean_tau < rho_v
                    rho_v = mean_tau
                end
            end

            rho_v = min(rho_max, rho_v)
            if I <= pq_end
                rho_v = max(rho_min_pq, rho_v)
            else
                rho_v = max(rho_min_w, rho_v)
            end
            rho[I] = rho_v
        end
    end

    return
end