__global__
void update_rho_kernel(int nvars, int k, int Kf, int Kf_mean, int pq_end,
                       double eps_rp, double eps_rp_min, double rt_inc, double rt_dec,
                       double eta, double rho_max, double rho_min_pq, double rho_min_w,
                       double *u_curr, double *u_prev,
                       double *v_curr, double *v_prev,
                       double *l_curr, double *l_prev,
                       double *rho, double *tau, double *rp, double *rp_old, double *rp_k0)
{
    int I = threadIdx.x + (blockDim.x * blockIdx.x);
    double delta_u, delta_v, delta_l, alpha, beta, rho_v, rp_v, rp_old_v, rp_k0_v, mean_tau;

    if (I < nvars) {
        delta_u = u_curr[I] - u_prev[I];
        delta_v = v_curr[I] - v_prev[I];
        delta_l = l_curr[I] - l_prev[I];
        alpha = abs(delta_l / delta_u);
        beta = abs(delta_l / delta_v);
        rho_v = rho[I];
        rp_v = rp[I];
        rp_old_v = rp_old[I];
        rp_k0_v = rp_k0[I];

        if (abs(delta_l) <= eps_rp_min) {
            tau[(k-1)*nvars + I] = tau[(k-2)*nvars + I];
        } else if (abs(delta_u) <= eps_rp_min && abs(delta_v) > eps_rp_min) {
            tau[(k-1)*nvars + I] = beta;
        } else if (abs(delta_u) > eps_rp_min && abs(delta_v) <= eps_rp_min) {
            tau[(k-1)*nvars + I] = alpha;
        } else if (abs(delta_u) <= eps_rp_min && abs(delta_v) <= eps_rp_min) {
            tau[(k-1)*nvars + I] = tau[(k-2)*nvars + I];
        } else {
            tau[(k-1)*nvars + I] = sqrt(alpha*beta);
        }

        if ((k % Kf) == 0) {
            mean_tau = 0;
            for (int j=0; j < Kf_mean; j++) {
                mean_tau += tau[(k-1-j)*nvars + I];
            }
            mean_tau /= Kf_mean;
            if (mean_tau >= rt_inc*rho_v) {
                if (abs(rp_v) >= eps_rp && abs(rp_old_v) >= eps_rp) {
                    if (abs(rp_v) > eta*abs(rp_k0_v) || abs(rp_old_v) > eta*abs(rp_k0_v)) {
                        rho_v *= rt_inc;
                    }
                }
            } else if (mean_tau > rho_v) {
                if (abs(rp_v) >= eps_rp && abs(rp_old_v) >= eps_rp) {
                    if (abs(rp_v) > eta*abs(rp_k0_v) || abs(rp_old_v) > eta*abs(rp_k0_v)) {
                        rho_v = mean_tau;
                    }
                }
            } else if (mean_tau <= rho_v/rt_dec) {
                rho_v /= rt_dec;
            } else if (mean_tau < rho_v) {
                rho_v = mean_tau;
            }
        }

        rho_v = min(rho_max, rho_v);
        if (I <= pq_end) {
            rho_v = max(rho_min_pq, rho_v);
        } else {
            rho_v = max(rho_min_w, rho_v);
        }
        rho[I] = rho_v;
    }

    return;
}
