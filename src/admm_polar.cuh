__device__
double eval_f_polar_kernel(int n, double scale, double *x, double *param,
                           double YffR, double YffI,
                           double YftR, double YftI,
                           double YttR, double YttI,
                           double YtfR, double YtfI)
{
    int I = blockIdx.x;
    int start = 31*I;
    double f = 0, vi_vj_cos, vi_vj_sin, pij, qij, pji, qji;

    vi_vj_cos = x[0]*x[1]*cos(x[2] - x[3]);
    vi_vj_sin = x[0]*x[1]*sin(x[2] - x[3]);
    pij = YffR*(x[0]*x[0]) + YftR*vi_vj_cos + YftI*vi_vj_sin;
    qij = -YffI*(x[0]*x[0]) - YftI*vi_vj_cos + YftR*vi_vj_sin;
    pji = YttR*(x[1]*x[1]) + YtfR*vi_vj_cos - YtfI*vi_vj_sin;
    qji = -YttI*(x[1]*x[1]) - YtfI*vi_vj_cos - YtfR*vi_vj_sin;

    f += param[start]*pij;
    f += param[start + 1]*qij;
    f += param[start + 2]*pji;
    f += param[start + 3]*qji;
    f += param[start + 4]*(x[0]*x[0]);
    f += param[start + 5]*(x[1]*x[1]);
    f += param[start + 6]*x[2];
    f += param[start + 7]*x[3];

    f += 0.5*(param[start + 8]*((pij - param[start + 16])*(pij - param[start + 16])));
    f += 0.5*(param[start + 9]*((qij - param[start + 17])*(qij - param[start + 17])));
    f += 0.5*(param[start + 10]*((pji - param[start + 18])*(pji - param[start + 18])));
    f += 0.5*(param[start + 11]*((qji - param[start + 19])*(qji - param[start + 19])));
    f += 0.5*(param[start + 12]*((x[0]*x[0] - param[start + 20])*(x[0]*x[0] - param[start + 20])));
    f += 0.5*(param[start + 13]*((x[1]*x[1] - param[start + 21])*(x[1]*x[1] - param[start + 21])));
    f += 0.5*(param[start + 14]*((x[2] - param[start + 22])*(x[2] - param[start + 22])));
    f += 0.5*(param[start + 15]*((x[3] - param[start + 23])*(x[3] - param[start + 23])));

    f *= scale;
    __syncthreads();

    return f;
}

__device__
void eval_grad_f_polar_kernel(int n, double scale, double *x, double *g, double *param,
                              double YffR, double YffI,
                              double YftR, double YftI,
                              double YttR, double YttI,
                              double YtfR, double YtfI)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int I = blockIdx.x;
    int start = 31*I;

    double cos_ij, sin_ij, vi_vj_cos, vi_vj_sin;
    double pij, qij, pji, qji;
    double g1, g2, g3, g4;
    double dpij_dx, dqij_dx, dpji_dx, dqji_dx;

    cos_ij = cos(x[2] - x[3]);
    sin_ij = sin(x[2] - x[3]);
    vi_vj_cos = x[0]*x[1]*cos_ij;
    vi_vj_sin = x[0]*x[1]*sin_ij;
    pij = YffR*(x[0]*x[0]) + YftR*vi_vj_cos + YftI*vi_vj_sin;
    qij = -YffI*(x[0]*x[0]) - YftI*vi_vj_cos + YftR*vi_vj_sin;
    pji = YttR*(x[1]*x[1]) + YtfR*vi_vj_cos - YtfI*vi_vj_sin;
    qji = -YttI*(x[1]*x[1]) - YtfI*vi_vj_cos - YtfR*vi_vj_sin;

    // Derivative with respect to vi.
    dpij_dx = 2*YffR*x[0] + YftR*x[1]*cos_ij + YftI*x[1]*sin_ij;
    dqij_dx = -2*YffI*x[0] - YftI*x[1]*cos_ij + YftR*x[1]*sin_ij;
    dpji_dx = YtfR*x[1]*cos_ij - YtfI*x[1]*sin_ij;
    dqji_dx = -YtfI*x[1]*cos_ij - YtfR*x[1]*sin_ij;

    g1 = param[start]*(dpij_dx);
    g1 += param[start + 1]*(dqij_dx);
    g1 += param[start + 2]*(dpji_dx);
    g1 += param[start + 3]*(dqji_dx);
    g1 += param[start + 4]*(2*x[0]);
    g1 += param[start + 8]*(pij - param[start + 16])*dpij_dx;
    g1 += param[start + 9]*(qij - param[start + 17])*dqij_dx;
    g1 += param[start + 10]*(pji - param[start + 18])*dpji_dx;
    g1 += param[start + 11]*(qji - param[start + 19])*dqji_dx;
    g1 += param[start + 12]*(x[0]*x[0] - param[start + 20])*(2*x[0]);

    // Derivative with respect to vj.
    dpij_dx = YftR*x[0]*cos_ij + YftI*x[0]*sin_ij;
    dqij_dx = -YftI*x[0]*cos_ij + YftR*x[0]*sin_ij;
    dpji_dx = 2*YttR*x[1] + YtfR*x[0]*cos_ij - YtfI*x[0]*sin_ij;
    dqji_dx = -2*YttI*x[1] - YtfI*x[0]*cos_ij - YtfR*x[0]*sin_ij;

    g2 = param[start]*(dpij_dx);
    g2 += param[start + 1]*(dqij_dx);
    g2 += param[start + 2]*(dpji_dx);
    g2 += param[start + 3]*(dqji_dx);
    g2 += param[start + 5]*(2*x[1]);
    g2 += param[start + 8]*(pij - param[start + 16])*dpij_dx;
    g2 += param[start + 9]*(qij - param[start + 17])*dqij_dx;
    g2 += param[start + 10]*(pji - param[start + 18])*dpji_dx;
    g2 += param[start + 11]*(qji - param[start + 19])*dqji_dx;
    g2 += param[start + 13]*(x[1]*x[1] - param[start + 21])*(2*x[1]);

    // Derivative with respect to ti.
    dpij_dx = -YftR*vi_vj_sin + YftI*vi_vj_cos;
    dqij_dx = YftI*vi_vj_sin + YftR*vi_vj_cos;
    dpji_dx = -YtfR*vi_vj_sin - YtfI*vi_vj_cos;
    dqji_dx = YtfI*vi_vj_sin - YtfR*vi_vj_cos;

    g3 = param[start]*(dpij_dx);
    g3 += param[start + 1]*(dqij_dx);
    g3 += param[start + 2]*(dpji_dx);
    g3 += param[start + 3]*(dqji_dx);
    g3 += param[start + 6];
    g3 += param[start + 8]*(pij - param[start + 16])*dpij_dx;
    g3 += param[start + 9]*(qij - param[start + 17])*dqij_dx;
    g3 += param[start + 10]*(pji - param[start + 18])*dpji_dx;
    g3 += param[start + 11]*(qji - param[start + 19])*dqji_dx;
    g3 += param[start + 14]*(x[2] - param[start + 22]);

    // Derivative with respect to tj.

    g4 = param[start]*(-dpij_dx);
    g4 += param[start + 1]*(-dqij_dx);
    g4 += param[start + 2]*(-dpji_dx);
    g4 += param[start + 3]*(-dqji_dx);
    g4 += param[start + 7];
    g4 += param[start + 8]*(pij - param[start + 16])*(-dpij_dx);
    g4 += param[start + 9]*(qij - param[start + 17])*(-dqij_dx);
    g4 += param[start + 10]*(pji - param[start + 18])*(-dpji_dx);
    g4 += param[start + 11]*(qji - param[start + 19])*(-dqji_dx);
    g4 += param[start + 15]*(x[3] - param[start + 23]);

    if (tx == 0 && ty == 0) {
        g[0] = scale*g1;
        g[1] = scale*g2;
        g[2] = scale*g3;
        g[3] = scale*g4;
    }

    __syncthreads();

    return;
}

__device__
void eval_h_polar_kernel(int n, double scale, double *x, double *A, double *param,
                         double YffR, double YffI,
                         double YftR, double YftI,
                         double YttR, double YttI,
                         double YtfR, double YtfI)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int start = 31*blockIdx.x;

    if (tx == 0 && ty == 0) {
        double cos_ij, sin_ij, vi_vj_cos, vi_vj_sin;
        double pij, qij, pji, qji;
        double dpij_dvi, dqij_dvi, dpji_dvi, dqji_dvi;
        double v;

        cos_ij = cos(x[2] - x[3]);
        sin_ij = sin(x[2] - x[3]);
        vi_vj_cos = x[0]*x[1]*cos_ij;
        vi_vj_sin = x[0]*x[1]*sin_ij;
        pij = YffR*(x[0]*x[0]) + YftR*vi_vj_cos + YftI*vi_vj_sin;
        qij = -YffI*(x[0]*x[0]) - YftI*vi_vj_cos + YftR*vi_vj_sin;
        pji = YttR*(x[1]*x[1]) + YtfR*vi_vj_cos - YtfI*vi_vj_sin;
        qji = -YttI*(x[1]*x[1]) - YtfI*vi_vj_cos - YtfR*vi_vj_sin;

        // d2f_dvidvi

        dpij_dvi = 2*YffR*x[0] + YftR*x[1]*cos_ij + YftI*x[1]*sin_ij;
        dqij_dvi = -2*YffI*x[0] - YftI*x[1]*cos_ij + YftR*x[1]*sin_ij;
        dpji_dvi = YtfR*x[1]*cos_ij - YtfI*x[1]*sin_ij;
        dqji_dvi = -YtfI*x[1]*cos_ij - YtfR*x[1]*sin_ij;

        // l_pij * d2pij_dvidvi
        v = param[start]*(2*YffR);
        // l_qij * d2qij_dvidvi
        v += param[start + 1]*(-2*YffI);
        // l_pji * d2pji_dvidvi = 0
        // l_qji * d2qji_dvidvi = 0
        // l_vi * 2
        v += 2*param[start + 4];
        // rho_pij*(dpij_dvi)^2 + rho_pij*(pij - tilde_pij)*d2pij_dvidvi
        v += param[start + 8]*(dpij_dvi*dpij_dvi) + param[start + 8]*(pij - param[start + 16])*(2*YffR);
        // rho_qij*(dqij_dvi)^2 + rho_qij*(qij - tilde_qij)*d2qij_dvidvi
        v += param[start + 9]*(dqij_dvi*dqij_dvi) + param[start + 9]*(qij - param[start + 17])*(-2*YffI);
        // rho_pji*(dpji_dvi)^2 + rho_pji*(pji - tilde_pji)*d2pji_dvidvi
        v += param[start + 10]*(dpji_dvi*dpji_dvi);
        // rho_qji*(dqji_dvi)^2
        v += param[start + 11]*(dqji_dvi*dqji_dvi);
        // (2*rho_vi*vi)*(2*vi) + rho_vi*(vi^2 - tilde_vi)*2
        v += 4*param[start + 12]*(x[0]*x[0]) + param[start + 12]*(x[0]*x[0] - param[start + 20])*2;
        A[0] = scale*v;

        // d2f_dvidvj

        double dpij_dvj, dqij_dvj, dpji_dvj, dqji_dvj;
        double d2pij_dvidvj, d2qij_dvidvj, d2pji_dvidvj, d2qji_dvidvj;

        dpij_dvj = YftR*x[0]*cos_ij + YftI*x[0]*sin_ij;
        dqij_dvj = -YftI*x[0]*cos_ij + YftR*x[0]*sin_ij;
        dpji_dvj = 2*YttR*x[1] + YtfR*x[0]*cos_ij - YtfI*x[0]*sin_ij;
        dqji_dvj = -2*YttI*x[1] - YtfI*x[0]*cos_ij - YtfR*x[0]*sin_ij;

        d2pij_dvidvj = YftR*cos_ij + YftI*sin_ij;
        d2qij_dvidvj = -YftI*cos_ij + YftR*sin_ij;
        d2pji_dvidvj = YtfR*cos_ij - YtfI*sin_ij;
        d2qji_dvidvj = -YtfI*cos_ij - YtfR*sin_ij;

        // l_pij * d2pij_dvidvj
        v = param[start]*(d2pij_dvidvj);
        // l_qij * d2qij_dvidvj
        v += param[start + 1]*(d2qij_dvidvj);
        // l_pji * d2pji_dvidvj
        v += param[start + 2]*(d2pji_dvidvj);
        // l_qji * d2qji_dvidvj
        v += param[start + 3]*(d2qji_dvidvj);
        // rho_pij*(dpij_dvj)*dpij_dvi + rho_pij*(pij - tilde_pij)*(d2pij_dvidvj)
        v += param[start + 8]*(dpij_dvj)*dpij_dvi + param[start + 8]*(pij - param[start + 16])*(d2pij_dvidvj);
        // rho_qij*(dqij_dvj)*dqij_dvi + rho_qij*(qij - tilde_qij)*(d2qij_dvidvj)
        v += param[start + 9]*(dqij_dvj)*dqij_dvi + param[start + 9]*(qij - param[start + 17])*(d2qij_dvidvj);
        // rho_pji*(dpji_dvj)*dpji_dvi + rho_pji*(pji - tilde_pji)*(d2pji_dvidvj)
        v += param[start + 10]*(dpji_dvj)*dpji_dvi + param[start + 10]*(pji - param[start + 18])*(d2pji_dvidvj);
        // rho_qji*(dqji_dvj)*dqji_dvi + rho_qji*(qji - tilde_qji)*(d2qji_dvidvj)
        v += param[start + 11]*(dqji_dvj)*dqji_dvi + param[start + 11]*(qji - param[start + 19])*(d2qji_dvidvj);
        A[1] = scale*v;

        // d2f_dvidti
        double dpij_dti, dqij_dti, dpji_dti, dqji_dti;
        double d2pij_dvidti, d2qij_dvidti, d2pji_dvidti, d2qji_dvidti;

        dpij_dti = -YftR*vi_vj_sin + YftI*vi_vj_cos;
        dqij_dti = YftI*vi_vj_sin + YftR*vi_vj_cos;
        dpji_dti = -YtfR*vi_vj_sin - YtfI*vi_vj_cos;
        dqji_dti = YtfI*vi_vj_sin - YtfR*vi_vj_cos;

        d2pij_dvidti = -YftR*x[1]*sin_ij + YftI*x[1]*cos_ij;
        d2qij_dvidti = YftI*x[1]*sin_ij + YftR*x[1]*cos_ij;
        d2pji_dvidti = -YtfR*x[1]*sin_ij - YtfI*x[1]*cos_ij;
        d2qji_dvidti = YtfI*x[1]*sin_ij - YtfR*x[1]*cos_ij;

        // l_pij * d2pij_dvidti
        v = param[start]*(d2pij_dvidti);
        // l_qij * d2qij_dvidti
        v += param[start + 1]*(d2qij_dvidti);
        // l_pji * d2pji_dvidti
        v += param[start + 2]*(d2pji_dvidti);
        // l_qji * d2qji_dvidti
        v += param[start + 3]*(d2qji_dvidti);
        // rho_pij*(dpij_dti)*dpij_dvi + rho_pij*(pij - tilde_pij)*(d2pij_dvidti)
        v += param[start + 8]*(dpij_dti)*dpij_dvi + param[start + 8]*(pij - param[start + 16])*(d2pij_dvidti);
        // rho_qij*(dqij_dti)*dqij_dvi + rho_qij*(qij - tilde_qij)*(d2qij_dvidti)
        v += param[start + 9]*(dqij_dti)*dqij_dvi + param[start + 9]*(qij - param[start + 17])*(d2qij_dvidti);
        // rho_pji*(dpji_dti)*dpji_dvi + rho_pji*(pji - tilde_pji)*(d2pji_dvidti)
        v += param[start + 10]*(dpji_dti)*dpji_dvi + param[start + 10]*(pji - param[start + 18])*(d2pji_dvidti);
        // rho_qji*(dqji_dti)*dqji_dvi + rho_qji*(qji - tilde_qji)*(d2qji_dvidti)
        v += param[start + 11]*(dqji_dti)*dqji_dvi + param[start + 11]*(qji - param[start + 19])*(d2qji_dvidti);
        A[2] = scale*v;

        // d2f_dvidtj

        // l_pij * d2pij_dvidtj
        v = param[start]*(-d2pij_dvidti);
        // l_qij * d2qij_dvidtj
        v += param[start + 1]*(-d2qij_dvidti);
        // l_pji * d2pji_dvidtj
        v += param[start + 2]*(-d2pji_dvidti);
        // l_qji * d2qji_dvidtj
        v += param[start + 3]*(-d2qji_dvidti);
        // rho_pij*(dpij_dtj)*dpij_dvi + rho_pij*(pij - tilde_pij)*(d2pij_dvidtj)
        v += param[start + 8]*(-dpij_dti)*dpij_dvi + param[start + 8]*(pij - param[start + 16])*(-d2pij_dvidti);
        // rho_qij*(dqij_dtj)*dqij_dvi + rho_qij*(qij - tilde_qij)*(d2qij_dvidtj)
        v += param[start + 9]*(-dqij_dti)*dqij_dvi + param[start + 9]*(qij - param[start + 17])*(-d2qij_dvidti);
        // rho_pji*(dpji_dtj)*dpji_dvi + rho_pji*(pji - tilde_pji)*(d2pji_dvidtj)
        v += param[start + 10]*(-dpji_dti)*dpji_dvi + param[start + 10]*(pji - param[start + 18])*(-d2pji_dvidti);
        // rho_qji*(dqji_dtj)*dqji_dvi + rho_qji*(qji - tilde_qji)*(d2qji_dvidtj)
        v += param[start + 11]*(-dqji_dti)*dqji_dvi + param[start + 11]*(qji - param[start + 19])*(-d2qji_dvidti);
        A[3] = scale*v;

        // d2f_dvjdvj

        // l_pij * d2pij_dvjdvj = l_qij * d2qij_dvjdvj = 0 since d2pij_dvjdvj = d2qij_dvjdvj = 0
        // l_pji * d2pji_dvjdvj
        v = param[start + 2]*(2*YttR);
        // l_qji * d2qji_dvjdvj
        v += param[start + 3]*(-2*YttI);
        // l_vj * 2
        v += param[start + 5]*2;
        // rho_pij*(dpij_dvj)^2 + rho_pij*(pij - tilde_pij)*(d2pij_dvjdvj)
        v += param[start + 8]*(dpij_dvj*dpij_dvj);
        // rho_qij*(dqij_dvj)^2 + rho_qij*(qij - tilde_qij)*(d2qij_dvjdvj)
        v += param[start + 9]*(dqij_dvj*dqij_dvj);
        // rho_pji*(dpji_dvj)^2 + rho_pji*(pji - tilde_pji)*(d2pji_dvjdvj)
        v += param[start + 10]*(dpji_dvj*dpji_dvj) + param[start + 10]*(pji - param[start + 18])*(2*YttR);
        // rho_qji*(dqji_dvj)^2 + rho_qji*(qji - tilde_qji)*(d2qji_dvjdvj)
        v += param[start + 11]*(dqji_dvj*dqji_dvj) + param[start + 11]*(qji - param[start + 19])*(-2*YttI);
        // (2*rho_vj*vj)*(2*vj) + rho_vj*(vj^2 - tilde_vj)*2
        v += 4*param[start + 13]*(x[1]*x[1]) + param[start + 13]*(x[1]*x[1] - param[start + 21])*2;
        A[n + 1] = scale*v;

        // d2f_dvjdti
        double d2pij_dvjdti, d2qij_dvjdti, d2pji_dvjdti, d2qji_dvjdti;

        d2pij_dvjdti = (-YftR*x[0]*sin_ij + YftI*x[0]*cos_ij);
        d2qij_dvjdti = (YftI*x[0]*sin_ij + YftR*x[0]*cos_ij);
        d2pji_dvjdti = (-YtfR*x[0]*sin_ij - YtfI*x[0]*cos_ij);
        d2qji_dvjdti = (YtfI*x[0]*sin_ij - YtfR*x[0]*cos_ij);

        // l_pij * d2pij_dvjdti
        v = param[start]*(d2pij_dvjdti);
        // l_qij * d2qij_dvjdti
        v += param[start + 1]*(d2qij_dvjdti);
        // l_pji * d2pji_dvjdti
        v += param[start + 2]*(d2pji_dvjdti);
        // l_qji * d2qji_dvjdti
        v += param[start + 3]*(d2qji_dvjdti);
        // rho_pij*(dpij_dti)*dpij_dvj + rho_pij*(pij - tilde_pij)*(d2pij_dvjdti)
        v += param[start + 8]*(dpij_dti)*dpij_dvj + param[start + 8]*(pij - param[start + 16])*d2pij_dvjdti;
        // rho_qij*(dqij_dti)*dqij_dvj + rho_qij*(qij - tilde_qij)*(d2qij_dvjdti)
        v += param[start + 9]*(dqij_dti)*dqij_dvj + param[start + 9]*(qij - param[start + 17])*d2qij_dvjdti;
        // rho_pji*(dpji_dti)*dpji_dvj + rho_pji*(pji - tilde_pji)*(d2pji_dvjdti)
        v += param[start + 10]*(dpji_dti)*dpji_dvj + param[start + 10]*(pji - param[start + 18])*d2pji_dvjdti;
        // rho_qji*(dqji_dti)*dqji_dvj + rho_qji*(qji - tilde_qji)*(d2qji_dvjdti)
        v += param[start + 11]*(dqji_dti)*dqji_dvj + param[start + 11]*(qji - param[start + 19])*d2qji_dvjdti;
        A[n + 2] = scale*v;

        // d2f_dvjdtj

        // l_pij * d2pij_dvjdtj
        v = param[start]*(-d2pij_dvjdti);
        // l_qij * d2qij_dvjdtj
        v += param[start + 1]*(-d2qij_dvjdti);
        // l_pji * d2pji_dvjdtj
        v += param[start + 2]*(-d2pji_dvjdti);
        // l_qji * d2qji_dvjdtj
        v += param[start + 3]*(-d2qji_dvjdti);
        // rho_pij*(dpij_dtj)*dpij_dvj + rho_pij*(pij - tilde_pij)*(d2pij_dvjdtj)
        v += param[start + 8]*(-dpij_dti)*dpij_dvj + param[start + 8]*(pij - param[start + 16])*(-d2pij_dvjdti);
        // rho_qij*(dqij_dtj)*dqij_dvj + rho_qij*(qij - tilde_qij)*(d2qij_dvjdtj)
        v += param[start + 9]*(-dqij_dti)*dqij_dvj + param[start + 9]*(qij - param[start + 17])*(-d2qij_dvjdti);
        // rho_pji*(dpji_dtj)*dpji_dvj + rho_pji*(pji - tilde_pji)*(d2pji_dvjdtj)
        v += param[start + 10]*(-dpji_dti)*dpji_dvj + param[start + 10]*(pji - param[start + 18])*(-d2pji_dvjdti);
        // rho_qji*(dqji_dtj)*dqji_dvj + rho_qji*(qji - tilde_qji)*(d2qji_dvjdtj)
        v += param[start + 11]*(-dqji_dti)*dqji_dvj + param[start + 11]*(qji - param[start + 19])*(-d2qji_dvjdti);
        A[n + 3] = scale*v;

        // d2f_dtidti
        double d2pij_dtidti, d2qij_dtidti, d2pji_dtidti, d2qji_dtidti;

        d2pij_dtidti = (-YftR*vi_vj_cos - YftI*vi_vj_sin);
        d2qij_dtidti = (YftI*vi_vj_cos - YftR*vi_vj_sin);
        d2pji_dtidti = (-YtfR*vi_vj_cos + YtfI*vi_vj_sin);
        d2qji_dtidti = (YtfI*vi_vj_cos + YtfR*vi_vj_sin);

        // l_pij * d2pij_dtidti
        v = param[start]*(d2pij_dtidti);
        // l_qij * d2qij_dtidti
        v += param[start + 1]*(d2qij_dtidti);
        // l_pji * d2pji_dtidti
        v += param[start + 2]*(d2pji_dtidti);
        // l_qji * d2qji_dtidti
        v += param[start + 3]*(d2qji_dtidti);
        // rho_pij*(dpij_dti)^2 + rho_pij*(pij - tilde_pij)*(d2pij_dtidti)
        v += param[start + 8]*(dpij_dti*dpij_dti) + param[start + 8]*(pij - param[start + 16])*(d2pij_dtidti);
        // rho_qij*(dqij_dti)^2 + rho_qij*(qij - tilde_qij)*(d2qij_dtidti)
        v += param[start + 9]*(dqij_dti*dqij_dti) + param[start + 9]*(qij - param[start + 17])*(d2qij_dtidti);
        // rho_pji*(dpji_dti)^2 + rho_pji*(pji - tilde_pji)*(d2pji_dtidti)
        v += param[start + 10]*(dpji_dti*dpji_dti) + param[start + 10]*(pji - param[start + 18])*(d2pji_dtidti);
        // rho_qji*(dqji_dti)^2 + rho_qji*(qji - tilde_qji)*(d2qji_dtidti)
        v += param[start + 11]*(dqji_dti*dqji_dti) + param[start + 11]*(qji - param[start + 19])*(d2qji_dtidti);
        // rho_ti
        v += param[start + 14];
        A[n*2 + 2] = scale*v;

        // d2f_dtidtj

        // l_pij * d2pij_dtidtj
        v = param[start]*(-d2pij_dtidti);
        // l_qij * d2qij_dtidtj
        v += param[start + 1]*(-d2qij_dtidti);
        // l_pji * d2pji_dtidtj
        v += param[start + 2]*(-d2pji_dtidti);
        // l_qji * d2qji_dtidtj
        v += param[start + 3]*(-d2qji_dtidti);
        // rho_pij*(dpij_dtj)*dpij_dti + rho_pij*(pij - tilde_pij)*(d2pij_dtidtj)
        v += param[start + 8]*(-(dpij_dti*dpij_dti)) + param[start + 8]*(pij - param[start + 16])*(-d2pij_dtidti);
        // rho_qij*(dqij_dtj)*dqij_dti + rho_qij*(qij - tilde_qij)*(d2qij_dtidtj)
        v += param[start + 9]*(-(dqij_dti*dqij_dti)) + param[start + 9]*(qij - param[start + 17])*(-d2qij_dtidti);
        // rho_pji*(dpji_dtj)*dpji_dti + rho_pji*(pji - tilde_pji)*(d2pji_dtidtj)
        v += param[start + 10]*(-(dpji_dti*dpji_dti)) + param[start + 10]*(pji - param[start + 18])*(-d2pji_dtidti);
        // rho_qji*(dqji_dtj)*dqji_dti + rho_qji*(qji - tilde_qji)*(d2qji_dtidtj)
        v += param[start + 11]*(-(dqji_dti*dqji_dti)) + param[start + 11]*(qji - param[start + 19])*(-d2qji_dtidti);
        A[n*2 + 3] = scale*v;

        // d2f_dtjdtj

        // l_pij * d2pij_dtjdtj
        v = param[start]*(d2pij_dtidti);
        // l_qij * d2qij_dtjdtj
        v += param[start + 1]*(d2qij_dtidti);
        // l_pji * d2pji_dtjdtj
        v += param[start + 2]*(d2pji_dtidti);
        // l_qji * d2qji_dtjdtj
        v += param[start + 3]*(d2qji_dtidti);
        // rho_pij*(dpij_dtj)^2 + rho_pij*(pij - tilde_pij)*(d2pij_dtjdtj)
        v += param[start + 8]*(dpij_dti*dpij_dti) + param[start + 8]*(pij - param[start + 16])*(d2pij_dtidti);
        // rho_qij*(dqij_dtj)^2 + rho_qij*(qij - tilde_qij)*(d2qij_dtjdtj)
        v += param[start + 9]*(dqij_dti*dqij_dti) + param[start + 9]*(qij - param[start + 17])*(d2qij_dtidti);
        // rho_pji*(dpji_dtj)^2 + rho_pji*(pji - tilde_pji)*(d2pji_dtjdtj)
        v += param[start + 10]*(dpji_dti*dpji_dti) + param[start + 10]*(pji - param[start + 18])*(d2pji_dtidti);
        // rho_qji*(dqji_dtj)^2 + rho_qji*(qji - tilde_qji)*(d2qji_dtjdtj)
        v += param[start + 11]*(dqji_dti*dqji_dti) + param[start + 11]*(qji - param[start + 19])*(d2qji_dtidti);
        // rho_tj
        v += param[start + 15];
        A[n*3 + 3] = scale*v;
    }

    __syncthreads();

    if (tx < n && ty == 0) {
        #pragma unroll
        for (int j = 0; j < n; j++) {
            if (tx > j) {
                A[n*tx + j] = A[n*j + tx];
            }
        }
    }

    __syncthreads();

    return;
}

__global__ void
//__launch_bounds__(32, 16)
polar_kernel(int nbranches, int n, int line_start, double scale,
             double *u_curr, double *v_curr, double *l_curr,
             double *rho, double *param,
             double *_YffR, double *_YffI,
             double *_YftR, double *_YftI,
             double *_YttR, double *_YttI,
             double *_YtfR, double *_YtfI,
             double *frBound, double *toBound)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int I = blockIdx.x;

    double YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI;

    extern __shared__ double shmem[];
    double *x, *xl, *xu;

    x = shmem;
    xl = shmem + n;
    xu = shmem + 2*n;

    int pij_idx = line_start + 8*I;

    if (tx == 0 && ty == 0) {
        xl[0] = sqrt(frBound[2*I]);
        xu[0] = sqrt(frBound[2*I+1]);
        xl[1] = sqrt(toBound[2*I]);
        xu[1] = sqrt(toBound[2*I+1]);
        xl[2] = -2*M_PI;
        xu[2] = 2*M_PI;
        xl[3] = -2*M_PI;
        xu[3] = 2*M_PI;

        x[0] = min(xu[0], max(xl[0], sqrt(u_curr[pij_idx+4])));
        x[1] = min(xu[1], max(xl[1], sqrt(u_curr[pij_idx+5])));
        x[2] = min(xu[2], max(xl[2], u_curr[pij_idx+6]));
        x[3] = min(xu[3], max(xl[3], u_curr[pij_idx+7]));
    }

    YffR = _YffR[I]; YffI = _YffI[I];
    YftR = _YftR[I]; YftI = _YftI[I];
    YttR = _YttR[I]; YttI = _YttI[I];
    YtfR = _YtfR[I]; YtfI = _YtfI[I];

    int start = 31*I;

    param[start] = l_curr[pij_idx];
    param[start + 1] = l_curr[pij_idx+1];
    param[start + 2] = l_curr[pij_idx+2];
    param[start + 3] = l_curr[pij_idx+3];
    param[start + 4] = l_curr[pij_idx+4];
    param[start + 5] = l_curr[pij_idx+5];
    param[start + 6] = l_curr[pij_idx+6];
    param[start + 7] = l_curr[pij_idx+7];
    param[start + 8] = rho[pij_idx];
    param[start + 9] = rho[pij_idx+1];
    param[start + 10] = rho[pij_idx+2];
    param[start + 11] = rho[pij_idx+3];
    param[start + 12] = rho[pij_idx+4];
    param[start + 13] = rho[pij_idx+5];
    param[start + 14] = rho[pij_idx+6];
    param[start + 15] = rho[pij_idx+7];
    param[start + 16] = v_curr[pij_idx];
    param[start + 17] = v_curr[pij_idx+1];
    param[start + 18] = v_curr[pij_idx+2];
    param[start + 19] = v_curr[pij_idx+3];
    param[start + 20] = v_curr[pij_idx+4];
    param[start + 21] = v_curr[pij_idx+5];
    param[start + 22] = v_curr[pij_idx+6];
    param[start + 23] = v_curr[pij_idx+7];

    __syncthreads();

    // Solve the branch problem.
    int status, minor_iter;

    cdriver_auglag(n, 500, 200, &status, &minor_iter, scale,
                    x, xl, xu, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI,
                    &eval_f_polar_kernel, &eval_grad_f_polar_kernel, &eval_h_polar_kernel);

    double vi_vj_cos = x[0]*x[1]*cos(x[2] - x[3]);
    double vi_vj_sin = x[0]*x[1]*sin(x[2] - x[3]);

    u_curr[pij_idx] = YffR*(x[0]*x[0]) + YftR*vi_vj_cos + YftI*vi_vj_sin;
    u_curr[pij_idx+1] = -YffI*(x[0]*x[0]) - YftI*vi_vj_cos + YftR*vi_vj_sin;
    u_curr[pij_idx+2] = YttR*(x[1]*x[1]) + YtfR*vi_vj_cos - YtfI*vi_vj_sin;
    u_curr[pij_idx+3] = -YttI*(x[1]*x[1]) - YtfI*vi_vj_cos - YtfR*vi_vj_sin;
    u_curr[pij_idx+4] = x[0]*x[0];
    u_curr[pij_idx+5] = x[1]*x[1];
    u_curr[pij_idx+6] = x[2];
    u_curr[pij_idx+7] = x[3];

    return;
}

double eval_f_polar(int I, int n, double scale, double *x, double *param,
                    double YffR, double YffI,
                    double YftR, double YftI,
                    double YttR, double YttI,
                    double YtfR, double YtfI)
{
    int start = 31*I;
    double f = 0, vi_vj_cos, vi_vj_sin, pij, qij, pji, qji;

    vi_vj_cos = x[0]*x[1]*cos(x[2] - x[3]);
    vi_vj_sin = x[0]*x[1]*sin(x[2] - x[3]);
    pij = YffR*(x[0]*x[0]) + YftR*vi_vj_cos + YftI*vi_vj_sin;
    qij = -YffI*(x[0]*x[0]) - YftI*vi_vj_cos + YftR*vi_vj_sin;
    pji = YttR*(x[1]*x[1]) + YtfR*vi_vj_cos - YtfI*vi_vj_sin;
    qji = -YttI*(x[1]*x[1]) - YtfI*vi_vj_cos - YtfR*vi_vj_sin;

    f += param[start]*pij;
    f += param[start + 1]*qij;
    f += param[start + 2]*pji;
    f += param[start + 3]*qji;
    f += param[start + 4]*(x[0]*x[0]);
    f += param[start + 5]*(x[1]*x[1]);
    f += param[start + 6]*x[2];
    f += param[start + 7]*x[3];

    f += 0.5*(param[start + 8]*((pij - param[start + 16])*(pij - param[start + 16])));
    f += 0.5*(param[start + 9]*((qij - param[start + 17])*(qij - param[start + 17])));
    f += 0.5*(param[start + 10]*((pji - param[start + 18])*(pji - param[start + 18])));
    f += 0.5*(param[start + 11]*((qji - param[start + 19])*(qji - param[start + 19])));
    f += 0.5*(param[start + 12]*((x[0]*x[0] - param[start + 20])*(x[0]*x[0] - param[start + 20])));
    f += 0.5*(param[start + 13]*((x[1]*x[1] - param[start + 21])*(x[1]*x[1] - param[start + 21])));
    f += 0.5*(param[start + 14]*((x[2] - param[start + 22])*(x[2] - param[start + 22])));
    f += 0.5*(param[start + 15]*((x[3] - param[start + 23])*(x[3] - param[start + 23])));

    f *= scale;

    return f;
}

void eval_grad_f_polar(int I, int n, double scale, double *x, double *g, double *param,
                       double YffR, double YffI,
                       double YftR, double YftI,
                       double YttR, double YttI,
                       double YtfR, double YtfI)
{
    int start = 31*I;

    double cos_ij, sin_ij, vi_vj_cos, vi_vj_sin;
    double pij, qij, pji, qji;
    double g1, g2, g3, g4;
    double dpij_dx, dqij_dx, dpji_dx, dqji_dx;

    cos_ij = cos(x[2] - x[3]);
    sin_ij = sin(x[2] - x[3]);
    vi_vj_cos = x[0]*x[1]*cos_ij;
    vi_vj_sin = x[0]*x[1]*sin_ij;
    pij = YffR*(x[0]*x[0]) + YftR*vi_vj_cos + YftI*vi_vj_sin;
    qij = -YffI*(x[0]*x[0]) - YftI*vi_vj_cos + YftR*vi_vj_sin;
    pji = YttR*(x[1]*x[1]) + YtfR*vi_vj_cos - YtfI*vi_vj_sin;
    qji = -YttI*(x[1]*x[1]) - YtfI*vi_vj_cos - YtfR*vi_vj_sin;

    // Derivative with respect to vi.
    dpij_dx = 2*YffR*x[0] + YftR*x[1]*cos_ij + YftI*x[1]*sin_ij;
    dqij_dx = -2*YffI*x[0] - YftI*x[1]*cos_ij + YftR*x[1]*sin_ij;
    dpji_dx = YtfR*x[1]*cos_ij - YtfI*x[1]*sin_ij;
    dqji_dx = -YtfI*x[1]*cos_ij - YtfR*x[1]*sin_ij;

    g1 = param[start]*(dpij_dx);
    g1 += param[start + 1]*(dqij_dx);
    g1 += param[start + 2]*(dpji_dx);
    g1 += param[start + 3]*(dqji_dx);
    g1 += param[start + 4]*(2*x[0]);
    g1 += param[start + 8]*(pij - param[start + 16])*dpij_dx;
    g1 += param[start + 9]*(qij - param[start + 17])*dqij_dx;
    g1 += param[start + 10]*(pji - param[start + 18])*dpji_dx;
    g1 += param[start + 11]*(qji - param[start + 19])*dqji_dx;
    g1 += param[start + 12]*(x[0]*x[0] - param[start + 20])*(2*x[0]);

    // Derivative with respect to vj.
    dpij_dx = YftR*x[0]*cos_ij + YftI*x[0]*sin_ij;
    dqij_dx = -YftI*x[0]*cos_ij + YftR*x[0]*sin_ij;
    dpji_dx = 2*YttR*x[1] + YtfR*x[0]*cos_ij - YtfI*x[0]*sin_ij;
    dqji_dx = -2*YttI*x[1] - YtfI*x[0]*cos_ij - YtfR*x[0]*sin_ij;

    g2 = param[start]*(dpij_dx);
    g2 += param[start + 1]*(dqij_dx);
    g2 += param[start + 2]*(dpji_dx);
    g2 += param[start + 3]*(dqji_dx);
    g2 += param[start + 5]*(2*x[1]);
    g2 += param[start + 8]*(pij - param[start + 16])*dpij_dx;
    g2 += param[start + 9]*(qij - param[start + 17])*dqij_dx;
    g2 += param[start + 10]*(pji - param[start + 18])*dpji_dx;
    g2 += param[start + 11]*(qji - param[start + 19])*dqji_dx;
    g2 += param[start + 13]*(x[1]*x[1] - param[start + 21])*(2*x[1]);

    // Derivative with respect to ti.
    dpij_dx = -YftR*vi_vj_sin + YftI*vi_vj_cos;
    dqij_dx = YftI*vi_vj_sin + YftR*vi_vj_cos;
    dpji_dx = -YtfR*vi_vj_sin - YtfI*vi_vj_cos;
    dqji_dx = YtfI*vi_vj_sin - YtfR*vi_vj_cos;

    g3 = param[start]*(dpij_dx);
    g3 += param[start + 1]*(dqij_dx);
    g3 += param[start + 2]*(dpji_dx);
    g3 += param[start + 3]*(dqji_dx);
    g3 += param[start + 6];
    g3 += param[start + 8]*(pij - param[start + 16])*dpij_dx;
    g3 += param[start + 9]*(qij - param[start + 17])*dqij_dx;
    g3 += param[start + 10]*(pji - param[start + 18])*dpji_dx;
    g3 += param[start + 11]*(qji - param[start + 19])*dqji_dx;
    g3 += param[start + 14]*(x[2] - param[start + 22]);

    // Derivative with respect to tj.

    g4 = param[start]*(-dpij_dx);
    g4 += param[start + 1]*(-dqij_dx);
    g4 += param[start + 2]*(-dpji_dx);
    g4 += param[start + 3]*(-dqji_dx);
    g4 += param[start + 7];
    g4 += param[start + 8]*(pij - param[start + 16])*(-dpij_dx);
    g4 += param[start + 9]*(qij - param[start + 17])*(-dqij_dx);
    g4 += param[start + 10]*(pji - param[start + 18])*(-dpji_dx);
    g4 += param[start + 11]*(qji - param[start + 19])*(-dqji_dx);
    g4 += param[start + 15]*(x[3] - param[start + 23]);

    g[0] = scale*g1;
    g[1] = scale*g2;
    g[2] = scale*g3;
    g[3] = scale*g4;

    return;
}

void eval_h_polar(int I, int n, double scale, double *x, double *A, double *param,
                  double YffR, double YffI,
                  double YftR, double YftI,
                  double YttR, double YttI,
                  double YtfR, double YtfI)
{
    int start = 31*I;

    double cos_ij, sin_ij, vi_vj_cos, vi_vj_sin;
    double pij, qij, pji, qji;
    double dpij_dvi, dqij_dvi, dpji_dvi, dqji_dvi;
    double v;

    cos_ij = cos(x[2] - x[3]);
    sin_ij = sin(x[2] - x[3]);
    vi_vj_cos = x[0]*x[1]*cos_ij;
    vi_vj_sin = x[0]*x[1]*sin_ij;
    pij = YffR*(x[0]*x[0]) + YftR*vi_vj_cos + YftI*vi_vj_sin;
    qij = -YffI*(x[0]*x[0]) - YftI*vi_vj_cos + YftR*vi_vj_sin;
    pji = YttR*(x[1]*x[1]) + YtfR*vi_vj_cos - YtfI*vi_vj_sin;
    qji = -YttI*(x[1]*x[1]) - YtfI*vi_vj_cos - YtfR*vi_vj_sin;

    // d2f_dvidvi

    dpij_dvi = 2*YffR*x[0] + YftR*x[1]*cos_ij + YftI*x[1]*sin_ij;
    dqij_dvi = -2*YffI*x[0] - YftI*x[1]*cos_ij + YftR*x[1]*sin_ij;
    dpji_dvi = YtfR*x[1]*cos_ij - YtfI*x[1]*sin_ij;
    dqji_dvi = -YtfI*x[1]*cos_ij - YtfR*x[1]*sin_ij;

    // l_pij * d2pij_dvidvi
    v = param[start]*(2*YffR);
    // l_qij * d2qij_dvidvi
    v += param[start + 1]*(-2*YffI);
    // l_pji * d2pji_dvidvi = 0
    // l_qji * d2qji_dvidvi = 0
    // l_vi * 2
    v += 2*param[start + 4];
    // rho_pij*(dpij_dvi)^2 + rho_pij*(pij - tilde_pij)*d2pij_dvidvi
    v += param[start + 8]*(dpij_dvi*dpij_dvi) + param[start + 8]*(pij - param[start + 16])*(2*YffR);
    // rho_qij*(dqij_dvi)^2 + rho_qij*(qij - tilde_qij)*d2qij_dvidvi
    v += param[start + 9]*(dqij_dvi*dqij_dvi) + param[start + 9]*(qij - param[start + 17])*(-2*YffI);
    // rho_pji*(dpji_dvi)^2 + rho_pji*(pji - tilde_pji)*d2pji_dvidvi
    v += param[start + 10]*(dpji_dvi*dpji_dvi);
    // rho_qji*(dqji_dvi)^2
    v += param[start + 11]*(dqji_dvi*dqji_dvi);
    // (2*rho_vi*vi)*(2*vi) + rho_vi*(vi^2 - tilde_vi)*2
    v += 4*param[start + 12]*(x[0]*x[0]) + param[start + 12]*(x[0]*x[0] - param[start + 20])*2;
    A[0] = scale*v;

    // d2f_dvidvj

    double dpij_dvj, dqij_dvj, dpji_dvj, dqji_dvj;
    double d2pij_dvidvj, d2qij_dvidvj, d2pji_dvidvj, d2qji_dvidvj;

    dpij_dvj = YftR*x[0]*cos_ij + YftI*x[0]*sin_ij;
    dqij_dvj = -YftI*x[0]*cos_ij + YftR*x[0]*sin_ij;
    dpji_dvj = 2*YttR*x[1] + YtfR*x[0]*cos_ij - YtfI*x[0]*sin_ij;
    dqji_dvj = -2*YttI*x[1] - YtfI*x[0]*cos_ij - YtfR*x[0]*sin_ij;

    d2pij_dvidvj = YftR*cos_ij + YftI*sin_ij;
    d2qij_dvidvj = -YftI*cos_ij + YftR*sin_ij;
    d2pji_dvidvj = YtfR*cos_ij - YtfI*sin_ij;
    d2qji_dvidvj = -YtfI*cos_ij - YtfR*sin_ij;

    // l_pij * d2pij_dvidvj
    v = param[start]*(d2pij_dvidvj);
    // l_qij * d2qij_dvidvj
    v += param[start + 1]*(d2qij_dvidvj);
    // l_pji * d2pji_dvidvj
    v += param[start + 2]*(d2pji_dvidvj);
    // l_qji * d2qji_dvidvj
    v += param[start + 3]*(d2qji_dvidvj);
    // rho_pij*(dpij_dvj)*dpij_dvi + rho_pij*(pij - tilde_pij)*(d2pij_dvidvj)
    v += param[start + 8]*(dpij_dvj)*dpij_dvi + param[start + 8]*(pij - param[start + 16])*(d2pij_dvidvj);
    // rho_qij*(dqij_dvj)*dqij_dvi + rho_qij*(qij - tilde_qij)*(d2qij_dvidvj)
    v += param[start + 9]*(dqij_dvj)*dqij_dvi + param[start + 9]*(qij - param[start + 17])*(d2qij_dvidvj);
    // rho_pji*(dpji_dvj)*dpji_dvi + rho_pji*(pji - tilde_pji)*(d2pji_dvidvj)
    v += param[start + 10]*(dpji_dvj)*dpji_dvi + param[start + 10]*(pji - param[start + 18])*(d2pji_dvidvj);
    // rho_qji*(dqji_dvj)*dqji_dvi + rho_qji*(qji - tilde_qji)*(d2qji_dvidvj)
    v += param[start + 11]*(dqji_dvj)*dqji_dvi + param[start + 11]*(qji - param[start + 19])*(d2qji_dvidvj);
    A[1] = scale*v;

    // d2f_dvidti
    double dpij_dti, dqij_dti, dpji_dti, dqji_dti;
    double d2pij_dvidti, d2qij_dvidti, d2pji_dvidti, d2qji_dvidti;

    dpij_dti = -YftR*vi_vj_sin + YftI*vi_vj_cos;
    dqij_dti = YftI*vi_vj_sin + YftR*vi_vj_cos;
    dpji_dti = -YtfR*vi_vj_sin - YtfI*vi_vj_cos;
    dqji_dti = YtfI*vi_vj_sin - YtfR*vi_vj_cos;

    d2pij_dvidti = -YftR*x[1]*sin_ij + YftI*x[1]*cos_ij;
    d2qij_dvidti = YftI*x[1]*sin_ij + YftR*x[1]*cos_ij;
    d2pji_dvidti = -YtfR*x[1]*sin_ij - YtfI*x[1]*cos_ij;
    d2qji_dvidti = YtfI*x[1]*sin_ij - YtfR*x[1]*cos_ij;

    // l_pij * d2pij_dvidti
    v = param[start]*(d2pij_dvidti);
    // l_qij * d2qij_dvidti
    v += param[start + 1]*(d2qij_dvidti);
    // l_pji * d2pji_dvidti
    v += param[start + 2]*(d2pji_dvidti);
    // l_qji * d2qji_dvidti
    v += param[start + 3]*(d2qji_dvidti);
    // rho_pij*(dpij_dti)*dpij_dvi + rho_pij*(pij - tilde_pij)*(d2pij_dvidti)
    v += param[start + 8]*(dpij_dti)*dpij_dvi + param[start + 8]*(pij - param[start + 16])*(d2pij_dvidti);
    // rho_qij*(dqij_dti)*dqij_dvi + rho_qij*(qij - tilde_qij)*(d2qij_dvidti)
    v += param[start + 9]*(dqij_dti)*dqij_dvi + param[start + 9]*(qij - param[start + 17])*(d2qij_dvidti);
    // rho_pji*(dpji_dti)*dpji_dvi + rho_pji*(pji - tilde_pji)*(d2pji_dvidti)
    v += param[start + 10]*(dpji_dti)*dpji_dvi + param[start + 10]*(pji - param[start + 18])*(d2pji_dvidti);
    // rho_qji*(dqji_dti)*dqji_dvi + rho_qji*(qji - tilde_qji)*(d2qji_dvidti)
    v += param[start + 11]*(dqji_dti)*dqji_dvi + param[start + 11]*(qji - param[start + 19])*(d2qji_dvidti);
    A[2] = scale*v;

    // d2f_dvidtj

    // l_pij * d2pij_dvidtj
    v = param[start]*(-d2pij_dvidti);
    // l_qij * d2qij_dvidtj
    v += param[start + 1]*(-d2qij_dvidti);
    // l_pji * d2pji_dvidtj
    v += param[start + 2]*(-d2pji_dvidti);
    // l_qji * d2qji_dvidtj
    v += param[start + 3]*(-d2qji_dvidti);
    // rho_pij*(dpij_dtj)*dpij_dvi + rho_pij*(pij - tilde_pij)*(d2pij_dvidtj)
    v += param[start + 8]*(-dpij_dti)*dpij_dvi + param[start + 8]*(pij - param[start + 16])*(-d2pij_dvidti);
    // rho_qij*(dqij_dtj)*dqij_dvi + rho_qij*(qij - tilde_qij)*(d2qij_dvidtj)
    v += param[start + 9]*(-dqij_dti)*dqij_dvi + param[start + 9]*(qij - param[start + 17])*(-d2qij_dvidti);
    // rho_pji*(dpji_dtj)*dpji_dvi + rho_pji*(pji - tilde_pji)*(d2pji_dvidtj)
    v += param[start + 10]*(-dpji_dti)*dpji_dvi + param[start + 10]*(pji - param[start + 18])*(-d2pji_dvidti);
    // rho_qji*(dqji_dtj)*dqji_dvi + rho_qji*(qji - tilde_qji)*(d2qji_dvidtj)
    v += param[start + 11]*(-dqji_dti)*dqji_dvi + param[start + 11]*(qji - param[start + 19])*(-d2qji_dvidti);
    A[3] = scale*v;

    // d2f_dvjdvj

    // l_pij * d2pij_dvjdvj = l_qij * d2qij_dvjdvj = 0 since d2pij_dvjdvj = d2qij_dvjdvj = 0
    // l_pji * d2pji_dvjdvj
    v = param[start + 2]*(2*YttR);
    // l_qji * d2qji_dvjdvj
    v += param[start + 3]*(-2*YttI);
    // l_vj * 2
    v += param[start + 5]*2;
    // rho_pij*(dpij_dvj)^2 + rho_pij*(pij - tilde_pij)*(d2pij_dvjdvj)
    v += param[start + 8]*(dpij_dvj*dpij_dvj);
    // rho_qij*(dqij_dvj)^2 + rho_qij*(qij - tilde_qij)*(d2qij_dvjdvj)
    v += param[start + 9]*(dqij_dvj*dqij_dvj);
    // rho_pji*(dpji_dvj)^2 + rho_pji*(pji - tilde_pji)*(d2pji_dvjdvj)
    v += param[start + 10]*(dpji_dvj*dpji_dvj) + param[start + 10]*(pji - param[start + 18])*(2*YttR);
    // rho_qji*(dqji_dvj)^2 + rho_qji*(qji - tilde_qji)*(d2qji_dvjdvj)
    v += param[start + 11]*(dqji_dvj*dqji_dvj) + param[start + 11]*(qji - param[start + 19])*(-2*YttI);
    // (2*rho_vj*vj)*(2*vj) + rho_vj*(vj^2 - tilde_vj)*2
    v += 4*param[start + 13]*(x[1]*x[1]) + param[start + 13]*(x[1]*x[1] - param[start + 21])*2;
    A[n + 1] = scale*v;

    // d2f_dvjdti
    double d2pij_dvjdti, d2qij_dvjdti, d2pji_dvjdti, d2qji_dvjdti;

    d2pij_dvjdti = (-YftR*x[0]*sin_ij + YftI*x[0]*cos_ij);
    d2qij_dvjdti = (YftI*x[0]*sin_ij + YftR*x[0]*cos_ij);
    d2pji_dvjdti = (-YtfR*x[0]*sin_ij - YtfI*x[0]*cos_ij);
    d2qji_dvjdti = (YtfI*x[0]*sin_ij - YtfR*x[0]*cos_ij);

    // l_pij * d2pij_dvjdti
    v = param[start]*(d2pij_dvjdti);
    // l_qij * d2qij_dvjdti
    v += param[start + 1]*(d2qij_dvjdti);
    // l_pji * d2pji_dvjdti
    v += param[start + 2]*(d2pji_dvjdti);
    // l_qji * d2qji_dvjdti
    v += param[start + 3]*(d2qji_dvjdti);
    // rho_pij*(dpij_dti)*dpij_dvj + rho_pij*(pij - tilde_pij)*(d2pij_dvjdti)
    v += param[start + 8]*(dpij_dti)*dpij_dvj + param[start + 8]*(pij - param[start + 16])*d2pij_dvjdti;
    // rho_qij*(dqij_dti)*dqij_dvj + rho_qij*(qij - tilde_qij)*(d2qij_dvjdti)
    v += param[start + 9]*(dqij_dti)*dqij_dvj + param[start + 9]*(qij - param[start + 17])*d2qij_dvjdti;
    // rho_pji*(dpji_dti)*dpji_dvj + rho_pji*(pji - tilde_pji)*(d2pji_dvjdti)
    v += param[start + 10]*(dpji_dti)*dpji_dvj + param[start + 10]*(pji - param[start + 18])*d2pji_dvjdti;
    // rho_qji*(dqji_dti)*dqji_dvj + rho_qji*(qji - tilde_qji)*(d2qji_dvjdti)
    v += param[start + 11]*(dqji_dti)*dqji_dvj + param[start + 11]*(qji - param[start + 19])*d2qji_dvjdti;
    A[n + 2] = scale*v;

    // d2f_dvjdtj

    // l_pij * d2pij_dvjdtj
    v = param[start]*(-d2pij_dvjdti);
    // l_qij * d2qij_dvjdtj
    v += param[start + 1]*(-d2qij_dvjdti);
    // l_pji * d2pji_dvjdtj
    v += param[start + 2]*(-d2pji_dvjdti);
    // l_qji * d2qji_dvjdtj
    v += param[start + 3]*(-d2qji_dvjdti);
    // rho_pij*(dpij_dtj)*dpij_dvj + rho_pij*(pij - tilde_pij)*(d2pij_dvjdtj)
    v += param[start + 8]*(-dpij_dti)*dpij_dvj + param[start + 8]*(pij - param[start + 16])*(-d2pij_dvjdti);
    // rho_qij*(dqij_dtj)*dqij_dvj + rho_qij*(qij - tilde_qij)*(d2qij_dvjdtj)
    v += param[start + 9]*(-dqij_dti)*dqij_dvj + param[start + 9]*(qij - param[start + 17])*(-d2qij_dvjdti);
    // rho_pji*(dpji_dtj)*dpji_dvj + rho_pji*(pji - tilde_pji)*(d2pji_dvjdtj)
    v += param[start + 10]*(-dpji_dti)*dpji_dvj + param[start + 10]*(pji - param[start + 18])*(-d2pji_dvjdti);
    // rho_qji*(dqji_dtj)*dqji_dvj + rho_qji*(qji - tilde_qji)*(d2qji_dvjdtj)
    v += param[start + 11]*(-dqji_dti)*dqji_dvj + param[start + 11]*(qji - param[start + 19])*(-d2qji_dvjdti);
    A[n + 3] = scale*v;

    // d2f_dtidti
    double d2pij_dtidti, d2qij_dtidti, d2pji_dtidti, d2qji_dtidti;

    d2pij_dtidti = (-YftR*vi_vj_cos - YftI*vi_vj_sin);
    d2qij_dtidti = (YftI*vi_vj_cos - YftR*vi_vj_sin);
    d2pji_dtidti = (-YtfR*vi_vj_cos + YtfI*vi_vj_sin);
    d2qji_dtidti = (YtfI*vi_vj_cos + YtfR*vi_vj_sin);

    // l_pij * d2pij_dtidti
    v = param[start]*(d2pij_dtidti);
    // l_qij * d2qij_dtidti
    v += param[start + 1]*(d2qij_dtidti);
    // l_pji * d2pji_dtidti
    v += param[start + 2]*(d2pji_dtidti);
    // l_qji * d2qji_dtidti
    v += param[start + 3]*(d2qji_dtidti);
    // rho_pij*(dpij_dti)^2 + rho_pij*(pij - tilde_pij)*(d2pij_dtidti)
    v += param[start + 8]*(dpij_dti*dpij_dti) + param[start + 8]*(pij - param[start + 16])*(d2pij_dtidti);
    // rho_qij*(dqij_dti)^2 + rho_qij*(qij - tilde_qij)*(d2qij_dtidti)
    v += param[start + 9]*(dqij_dti*dqij_dti) + param[start + 9]*(qij - param[start + 17])*(d2qij_dtidti);
    // rho_pji*(dpji_dti)^2 + rho_pji*(pji - tilde_pji)*(d2pji_dtidti)
    v += param[start + 10]*(dpji_dti*dpji_dti) + param[start + 10]*(pji - param[start + 18])*(d2pji_dtidti);
    // rho_qji*(dqji_dti)^2 + rho_qji*(qji - tilde_qji)*(d2qji_dtidti)
    v += param[start + 11]*(dqji_dti*dqji_dti) + param[start + 11]*(qji - param[start + 19])*(d2qji_dtidti);
    // rho_ti
    v += param[start + 14];
    A[n*2 + 2] = scale*v;

    // d2f_dtidtj

    // l_pij * d2pij_dtidtj
    v = param[start]*(-d2pij_dtidti);
    // l_qij * d2qij_dtidtj
    v += param[start + 1]*(-d2qij_dtidti);
    // l_pji * d2pji_dtidtj
    v += param[start + 2]*(-d2pji_dtidti);
    // l_qji * d2qji_dtidtj
    v += param[start + 3]*(-d2qji_dtidti);
    // rho_pij*(dpij_dtj)*dpij_dti + rho_pij*(pij - tilde_pij)*(d2pij_dtidtj)
    v += param[start + 8]*(-(dpij_dti*dpij_dti)) + param[start + 8]*(pij - param[start + 16])*(-d2pij_dtidti);
    // rho_qij*(dqij_dtj)*dqij_dti + rho_qij*(qij - tilde_qij)*(d2qij_dtidtj)
    v += param[start + 9]*(-(dqij_dti*dqij_dti)) + param[start + 9]*(qij - param[start + 17])*(-d2qij_dtidti);
    // rho_pji*(dpji_dtj)*dpji_dti + rho_pji*(pji - tilde_pji)*(d2pji_dtidtj)
    v += param[start + 10]*(-(dpji_dti*dpji_dti)) + param[start + 10]*(pji - param[start + 18])*(-d2pji_dtidti);
    // rho_qji*(dqji_dtj)*dqji_dti + rho_qji*(qji - tilde_qji)*(d2qji_dtidtj)
    v += param[start + 11]*(-(dqji_dti*dqji_dti)) + param[start + 11]*(qji - param[start + 19])*(-d2qji_dtidti);
    A[n*2 + 3] = scale*v;

    // d2f_dtjdtj

    // l_pij * d2pij_dtjdtj
    v = param[start]*(d2pij_dtidti);
    // l_qij * d2qij_dtjdtj
    v += param[start + 1]*(d2qij_dtidti);
    // l_pji * d2pji_dtjdtj
    v += param[start + 2]*(d2pji_dtidti);
    // l_qji * d2qji_dtjdtj
    v += param[start + 3]*(d2qji_dtidti);
    // rho_pij*(dpij_dtj)^2 + rho_pij*(pij - tilde_pij)*(d2pij_dtjdtj)
    v += param[start + 8]*(dpij_dti*dpij_dti) + param[start + 8]*(pij - param[start + 16])*(d2pij_dtidti);
    // rho_qij*(dqij_dtj)^2 + rho_qij*(qij - tilde_qij)*(d2qij_dtjdtj)
    v += param[start + 9]*(dqij_dti*dqij_dti) + param[start + 9]*(qij - param[start + 17])*(d2qij_dtidti);
    // rho_pji*(dpji_dtj)^2 + rho_pji*(pji - tilde_pji)*(d2pji_dtjdtj)
    v += param[start + 10]*(dpji_dti*dpji_dti) + param[start + 10]*(pji - param[start + 18])*(d2pji_dtidti);
    // rho_qji*(dqji_dtj)^2 + rho_qji*(qji - tilde_qji)*(d2qji_dtjdtj)
    v += param[start + 11]*(dqji_dti*dqji_dti) + param[start + 11]*(qji - param[start + 19])*(d2qji_dtidti);
    // rho_tj
    v += param[start + 15];
    A[n*3 + 3] = scale*v;

    #pragma unroll
    for (int j = 0; j < n; j++) {
        for (int k = 0; k < j; k++) {
            A[n*j + k] = A[n*k + j];
        }
    }

    return;
}

void polar(int nbranches, int n, int line_start, double scale,
           double *u_curr, double *v_curr, double *l_curr,
           double *rho, double *param,
           double *_YffR, double *_YffI,
           double *_YftR, double *_YftI,
           double *_YttR, double *_YttI,
           double *_YtfR, double *_YtfI,
           double *frBound, double *toBound)
{
    double YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI;
    double *x, *xl, *xu;

    x = (double *)calloc(n, sizeof(double));
    xl = (double *)calloc(n, sizeof(double));
    xu = (double *)calloc(n, sizeof(double));

    for (int I = 0; I < nbranches; I++) {
        int pij_idx = line_start + 8*I;

        xl[0] = sqrt(frBound[2*I]);
        xu[0] = sqrt(frBound[2*I+1]);
        xl[1] = sqrt(toBound[2*I]);
        xu[1] = sqrt(toBound[2*I+1]);
        xl[2] = -2*M_PI;
        xu[2] = 2*M_PI;
        xl[3] = -2*M_PI;
        xu[3] = 2*M_PI;

        x[0] = min(xu[0], max(xl[0], sqrt(u_curr[pij_idx+4])));
        x[1] = min(xu[1], max(xl[1], sqrt(u_curr[pij_idx+5])));
        x[2] = min(xu[2], max(xl[2], u_curr[pij_idx+6]));
        x[3] = min(xu[3], max(xl[3], u_curr[pij_idx+7]));

        YffR = _YffR[I]; YffI = _YffI[I];
        YftR = _YftR[I]; YftI = _YftI[I];
        YttR = _YttR[I]; YttI = _YttI[I];
        YtfR = _YtfR[I]; YtfI = _YtfI[I];

        int start = 31*I;

        param[start] = l_curr[pij_idx];
        param[start + 1] = l_curr[pij_idx+1];
        param[start + 2] = l_curr[pij_idx+2];
        param[start + 3] = l_curr[pij_idx+3];
        param[start + 4] = l_curr[pij_idx+4];
        param[start + 5] = l_curr[pij_idx+5];
        param[start + 6] = l_curr[pij_idx+6];
        param[start + 7] = l_curr[pij_idx+7];
        param[start + 8] = rho[pij_idx];
        param[start + 9] = rho[pij_idx+1];
        param[start + 10] = rho[pij_idx+2];
        param[start + 11] = rho[pij_idx+3];
        param[start + 12] = rho[pij_idx+4];
        param[start + 13] = rho[pij_idx+5];
        param[start + 14] = rho[pij_idx+6];
        param[start + 15] = rho[pij_idx+7];
        param[start + 16] = v_curr[pij_idx];
        param[start + 17] = v_curr[pij_idx+1];
        param[start + 18] = v_curr[pij_idx+2];
        param[start + 19] = v_curr[pij_idx+3];
        param[start + 20] = v_curr[pij_idx+4];
        param[start + 21] = v_curr[pij_idx+5];
        param[start + 22] = v_curr[pij_idx+6];
        param[start + 23] = v_curr[pij_idx+7];

        // Solve the branch problem.
        int status, minor_iter;

        driver_auglag(I, n, 500, 200, &status, &minor_iter, scale,
                      x, xl, xu, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI,
                      &eval_f_polar, &eval_grad_f_polar, &eval_h_polar);

        double vi_vj_cos = x[0]*x[1]*cos(x[2] - x[3]);
        double vi_vj_sin = x[0]*x[1]*sin(x[2] - x[3]);

        u_curr[pij_idx] = YffR*(x[0]*x[0]) + YftR*vi_vj_cos + YftI*vi_vj_sin;
        u_curr[pij_idx+1] = -YffI*(x[0]*x[0]) - YftI*vi_vj_cos + YftR*vi_vj_sin;
        u_curr[pij_idx+2] = YttR*(x[1]*x[1]) + YtfR*vi_vj_cos - YtfI*vi_vj_sin;
        u_curr[pij_idx+3] = -YttI*(x[1]*x[1]) - YtfI*vi_vj_cos - YtfR*vi_vj_sin;
        u_curr[pij_idx+4] = x[0]*x[0];
        u_curr[pij_idx+5] = x[1]*x[1];
        u_curr[pij_idx+6] = x[2];
        u_curr[pij_idx+7] = x[3];
    }

    free(x);
    free(xl);
    free(xu);

    return;
}
