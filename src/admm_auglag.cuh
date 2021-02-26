__device__
double eval_f_kernel(int n, double *x, double *param,
                     double YffR, double YffI,
                     double YftR, double YftI,
                     double YttR, double YttI,
                     double YtfR, double YtfI)
{
    int I = blockIdx.x;
    double f = 0, c1, c2, c3, c4, c5, raug;

    int start = 24*I;
    for (int j = 0; j < 6; j++) {
        f += param[start + j]*x[j];
    }
    for (int j = 0; j < 6; j++) {
        f += 0.5*(param[start + 6+j]*(x[j] - param[start + 12+j])*(x[j] - param[start + 12+j]));
    }

    c1 = (x[0] - (YffR*x[4] + YftR*x[6] + YftI*x[7]));
    c2 = (x[1] - (-YffI*x[4] - YftI*x[6] + YftR*x[7]));
    c3 = (x[2] - (YttR*x[5] + YtfR*x[6] - YtfI*x[7]));
    c4 = (x[3] - (-YttI*x[5] - YtfI*x[6] - YtfR*x[7]));
    c5 = (x[6]*x[6] + x[7]*x[7] - x[4]*x[5]);

    f += param[start + 18]*c1;
    f += param[start + 19]*c2;
    f += param[start + 20]*c3;
    f += param[start + 21]*c4;
    f += param[start + 22]*c5;

    raug = param[start + 23];
    f += 0.5*raug*(c1*c1);
    f += 0.5*raug*(c2*c2);
    f += 0.5*raug*(c3*c3);
    f += 0.5*raug*(c4*c4);
    f += 0.5*raug*(c5*c5);

    __syncthreads();
    return f;
}

__device__
void eval_grad_f_kernel(int n, double *x, double *g, double *param,
                        double YffR, double YffI,
                        double YftR, double YftI,
                        double YttR, double YttI,
                        double YtfR, double YtfI)
{
    int I = blockIdx.x;

    double c1, c2, c3, c4, c5;
    double g1, g2, g3, g4, g5, g6, g7, g8;
    double raug;

    c1 = (x[0] - (YffR*x[4] + YftR*x[6] + YftI*x[7]));
    c2 = (x[1] - (-YffI*x[4] - YftI*x[6] + YftR*x[7]));
    c3 = (x[2] - (YttR*x[5] + YtfR*x[6] - YtfI*x[7]));
    c4 = (x[3] - (-YttI*x[5] - YtfI*x[6] - YtfR*x[7]));
    c5 = (x[6]*x[6] + x[7]*x[7] - x[4]*x[5]);

    int start = 24*I;
    g1 = param[start] + param[start + 6]*(x[0] - param[start + 12]);
    g2 = param[start + 1] + param[start + 7]*(x[1] - param[start + 13]);
    g3 = param[start + 2] + param[start + 8]*(x[2] - param[start + 14]);
    g4 = param[start + 3] + param[start + 9]*(x[3] - param[start + 15]);
    g5 = param[start + 4] + param[start + 10]*(x[4] - param[start + 16]);
    g6 = param[start + 5] + param[start + 11]*(x[5] - param[start + 17]);


    raug = param[start + 23];
    g1 += param[start + 18] + raug*c1;
    g2 += param[start + 19] + raug*c2;
    g3 += param[start + 20] + raug*c3;
    g4 += param[start + 21] + raug*c4;

    g5 += param[start + 18]*(-YffR) + param[start + 19]*(YffI) + param[start + 22]*(-x[5]) +
          raug*(-YffR)*c1 + raug*(YffI)*c2 + raug*(-x[5])*c5;
    g6 += param[start + 20]*(-YttR) + param[start + 21]*(YttI) + param[start + 22]*(-x[4]) +
          raug*(-YttR)*c3 + raug*(YttI)*c4 + raug*(-x[4])*c5;
    g7 = param[start + 18]*(-YftR) + param[start + 19]*(YftI) + param[start + 20]*(-YtfR) +
         param[start + 21]*(YtfI) + param[start + 22]*(2*x[6]) +
         raug*(-YftR)*c1 + raug*(YftI)*c2 + raug*(-YtfR)*c3 +
         raug*(YtfI)*c4 + raug*(2*x[6])*c5;
    g8 = param[start + 18]*(-YftI) + param[start + 19]*(-YftR) + param[start + 20]*(YtfI) +
         param[start + 21]*(YtfR) + param[start + 22]*(2*x[7]) +
         raug*(-YftI)*c1 + raug*(-YftR)*c2 + raug*(YtfI)*c3 +
         raug*(YtfR)*c4 + raug*(2*x[7])*c5;

    g[0] = g1;
    g[1] = g2;
    g[2] = g3;
    g[3] = g4;
    g[4] = g5;
    g[5] = g6;
    g[6] = g7;
    g[7] = g8;

    __syncthreads();
    return;
}

__device__
void eval_h_kernel(int n, double *x, double *A, double *param,
                   double YffR, double YffI,
                   double YftR, double YftI,
                   double YttR, double YttI,
                   double YtfR, double YtfI)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int I = blockIdx.x;

    double alrect, raug, c5;
    int start = 24*I;

    alrect = param[start + 22];
    raug = param[start + 23];
    c5 = (x[6]*x[6] + x[7]*x[7] - x[4]*x[5]);

    // 1st column
    A[0] = param[start + 6] + raug; // A[1,1]
    A[4] = raug*(-YffR); // A[5,1]
    A[6] = raug*(-YftR); // A[7,1]
    A[7] = raug*(-YftI); // A[8,1]

    // 2nd columns
    A[n + 1] = param[start + 7] + raug; // A[2,2]
    A[n + 4] = raug*(YffI);  // A[5,2]
    A[n + 6] = raug*(YftI);  // A[7,2]
    A[n + 7] = raug*(-YftR); // A[8,2]

    // 3rd column
    A[n*2 + 2] = param[start + 8] + raug; // A[3,3]
    A[n*2 + 5] = raug*(-YttR); // A[6,3]
    A[n*2 + 6] = raug*(-YtfR); // A[7,3]
    A[n*2 + 7] = raug*(YtfI);  // A[8,3]

    // 4th column
    A[n*3 + 3] = param[start + 9] + raug; // A[4,4]
    A[n*3 + 5] = raug*(YttI); // A[6,4]
    A[n*3 + 6] = raug*(YtfI); // A[7,4]
    A[n*3 + 7] = raug*(YtfR); // A[8,4]

    // 5th column
    A[n*4 + 4] = param[start + 10] + raug*(YffR*YffR) + raug*(YffI*YffI) + raug*(x[5]*x[5]); // A[5,5]
    A[n*4 + 5] = -(alrect + raug*c5) + raug*(x[4]*x[5]); // A[6,5]
    A[n*4 + 6] = raug*(YffR*YftR) + raug*(YffI*YftI) + raug*((-x[5])*(2*x[6])); // A[7,5]
    A[n*4 + 7] = raug*(YffR*YftI) + raug*(YffI*(-YftR)) + raug*((-x[5])*(2*x[7])); // A[8,5]

    // 6th column
    A[n*5 + 5] = param[start + 11] + raug*(YttR*YttR) + raug*(YttI*YttI) + raug*(x[4]*x[4]); // A[6,6]
    A[n*5 + 6] = raug*(YttR*YtfR) + raug*(YttI*YtfI) + raug*((-x[4])*(2*x[6])); // A[7,6]
    A[n*5 + 7] = raug*((-YttR)*YtfI) + raug*(YttI*YtfR) + raug*((-x[4])*(2*x[7])); // A[8,6]

    // 7th column
    A[n*6 + 6] = (alrect + raug*c5)*2 + raug*(YftR*YftR) + raug*(YftI*YftI) +
                 raug*(YtfR*YtfR) + raug*(YtfI*YtfI) + raug*((2*x[6])*(2*x[6])); // A[7,7]
    A[n*6 + 7] = raug*(YftR*YftI) + raug*(YftI*(-YftR)) + raug*((-YtfR)*YtfI) +
                 raug*(YtfI*YtfR) + raug*((2*x[6])*(2*x[7])); // A[8,7]

    // 8th column
    A[n*7 + 7] = (alrect + raug*c5)*2 + raug*(YftI*YftI) + raug*(YftR*YftR) +
                 raug*(YtfI*YtfI) + raug*(YtfR*YtfR) + raug*((2*x[7])*(2*x[7])); // A[8,8]

    if (tx > ty) {
        A[n*tx + ty] = A[n*ty + tx];
    }

    __syncthreads();
    return;
}

__global__
void auglag_kernel(int nbranches, int n, int major_iter, int max_auglag,
                   int pij_start, int qij_start,
                   int pji_start, int qji_start,
                   int wi_i_ij_start, int wi_j_ji_start,
                   double mu_max,
                   double *u_curr, double *v_curr, double *l_curr,
                   double *rho, double *wRIij, double *param,
                   double *_YffR, double *_YffI,
                   double *_YftR, double *_YftI,
                   double *_YttR, double *_YttI,
                   double *_YtfR, double *_YtfI,
                   double *frBound, double *toBound)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int I = blockIdx.x;

    bool terminate;
    int max_feval, max_minor, it, status, minor_iter;
    double mu, eta, omega;
    double YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI;

    extern __shared__ double shmem[];
    double *x, *xl, *xu;

    x = shmem;
    xl = shmem + n;
    xu = shmem + 2*n;

    if (tx == 0 && ty == 0) {
        for (int j = 0; j < n; j++) {
            xl[j] = -INF;
            xu[j] = INF;
        }

        xl[4] = frBound[2*I];
        xu[4] = frBound[2*I+1];
        xl[5] = toBound[2*I];
        xu[5] = toBound[2*I+1];

        x[0] = u_curr[pij_start+I];
        x[1] = u_curr[qij_start+I];
        x[2] = u_curr[pji_start+I];
        x[3] = u_curr[qji_start+I];
        x[4] = min(xu[4], max(xl[4], u_curr[wi_i_ij_start+I]));
        x[5] = min(xu[5], max(xl[5], u_curr[wi_j_ji_start+I]));
        x[6] = wRIij[2*I];
        x[7] = wRIij[2*I+1];
    }

    YffR = _YffR[I]; YffI = _YffI[I];
    YftR = _YftR[I]; YftI = _YftI[I];
    YttR = _YttR[I]; YttI = _YttI[I];
    YtfR = _YtfR[I]; YtfI = _YtfI[I];

    int start = 24*I;
    param[start] = l_curr[pij_start+I];
    param[start + 1] = l_curr[qij_start+I];
    param[start + 2] = l_curr[pji_start+I];
    param[start + 3] = l_curr[qji_start+I];
    param[start + 4] = l_curr[wi_i_ij_start+I];
    param[start + 5] = l_curr[wi_j_ji_start+I];
    param[start + 6] = rho[pij_start+I];
    param[start + 7] = rho[qij_start+I];
    param[start + 8] = rho[pji_start+I];
    param[start + 9] = rho[qji_start+I];
    param[start + 10] = rho[wi_i_ij_start+I];
    param[start + 11] = rho[wi_j_ji_start+I];
    param[start + 12] = v_curr[pij_start+I];
    param[start + 13] = v_curr[qij_start+I];
    param[start + 14] = v_curr[pji_start+I];
    param[start + 15] = v_curr[qji_start+I];
    param[start + 16] = v_curr[wi_i_ij_start+I];
    param[start + 17] = v_curr[wi_j_ji_start+I];

    if (major_iter == 1) {
        param[start + 23] = 10.0;
        mu = 10.0;
    } else {
        mu = param[start + 23];
    }

    __syncthreads();

    eta = 1 / pow(mu, 0.1);
    omega = 1 / mu;
    max_feval = 500;
    max_minor = 200;

    it = 0;
    terminate = false;

    double cviol1, cviol2, cviol3, cviol4, cviol5, cnorm;
    while (!terminate) {
        it += 1;

        // Solve the branch problem.
        cdriver_auglag(n, max_feval, max_minor, &status, &minor_iter,
                       x, xl, xu, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI,
                       &eval_f_kernel, &eval_grad_f_kernel, &eval_h_kernel);

        // Check the termination condition.
        cviol1 = x[0] - (YffR*x[4] + YftR*x[6] + YftI*x[7]);
        cviol2 = x[1] - (-YffI*x[4] - YftI*x[6] + YftR*x[7]);
        cviol3 = x[2] - (YttR*x[5] + YtfR*x[6] - YtfI*x[7]);
        cviol4 = x[3] - (-YttI*x[5] - YtfI*x[6] - YtfR*x[7]);
        cviol5 = x[6]*x[6] + x[7]*x[7] - x[4]*x[5];

        cnorm = max(abs(cviol1), max(abs(cviol2), max(abs(cviol3), max(abs(cviol4), abs(cviol5)))));

        if (cnorm <= eta) {
            if (cnorm <= 1e-6) {
                terminate = true;
            } else {
                if (tx == 0 && ty == 0) {
                    param[start + 18] += mu*cviol1;
                    param[start + 19] += mu*cviol2;
                    param[start + 20] += mu*cviol3;
                    param[start + 21] += mu*cviol4;
                    param[start + 22] += mu*cviol5;

                    eta = eta / pow(mu, 0.9);
                    omega  = omega / mu;
                }
            }
        } else {
            mu = min(mu_max, mu*10);
            eta = 1 / pow(mu, 0.1);
            omega = 1 / mu;
            param[start + 23] = mu;
        }

        if (it >= max_auglag) {
            terminate = true;
        }

        __syncthreads();
    }

    u_curr[pij_start+I] = x[0];
    u_curr[qij_start+I] = x[1];
    u_curr[pji_start+I] = x[2];
    u_curr[qji_start+I] = x[3];
    u_curr[wi_i_ij_start+I] = x[4];
    u_curr[wi_j_ji_start+I] = x[5];
    wRIij[2*I] = x[6];
    wRIij[2*I+1] = x[7];
    param[start + 23] = mu;

    __syncthreads();

    return;
}

double eval_f(int I, int n, double *x, double *param,
              double YffR, double YffI,
              double YftR, double YftI,
              double YttR, double YttI,
              double YtfR, double YtfI)
{
    double f = 0, c1, c2, c3, c4, c5, raug;

    int start = 24*I;
    for (int j = 0; j < 6; j++) {
        f += param[start + j]*x[j];
    }
    for (int j = 0; j < 6; j++) {
        f += 0.5*(param[start + 6+j]*(x[j] - param[start + 12+j])*(x[j] - param[start + 12+j]));
    }

    c1 = (x[0] - (YffR*x[4] + YftR*x[6] + YftI*x[7]));
    c2 = (x[1] - (-YffI*x[4] - YftI*x[6] + YftR*x[7]));
    c3 = (x[2] - (YttR*x[5] + YtfR*x[6] - YtfI*x[7]));
    c4 = (x[3] - (-YttI*x[5] - YtfI*x[6] - YtfR*x[7]));
    c5 = (x[6]*x[6] + x[7]*x[7] - x[4]*x[5]);

    f += param[start + 18]*c1;
    f += param[start + 19]*c2;
    f += param[start + 20]*c3;
    f += param[start + 21]*c4;
    f += param[start + 22]*c5;

    raug = param[start + 23];
    f += 0.5*raug*(c1*c1);
    f += 0.5*raug*(c2*c2);
    f += 0.5*raug*(c3*c3);
    f += 0.5*raug*(c4*c4);
    f += 0.5*raug*(c5*c5);

    return f;
}

void eval_grad_f(int I, int n, double *x, double *g, double *param,
                double YffR, double YffI,
                double YftR, double YftI,
                double YttR, double YttI,
                double YtfR, double YtfI)
{
    double c1, c2, c3, c4, c5;
    double g1, g2, g3, g4, g5, g6, g7, g8;
    double raug;

    c1 = (x[0] - (YffR*x[4] + YftR*x[6] + YftI*x[7]));
    c2 = (x[1] - (-YffI*x[4] - YftI*x[6] + YftR*x[7]));
    c3 = (x[2] - (YttR*x[5] + YtfR*x[6] - YtfI*x[7]));
    c4 = (x[3] - (-YttI*x[5] - YtfI*x[6] - YtfR*x[7]));
    c5 = (x[6]*x[6] + x[7]*x[7] - x[4]*x[5]);

    int start = 24*I;
    g1 = param[start] + param[start + 6]*(x[0] - param[start + 12]);
    g2 = param[start + 1] + param[start + 7]*(x[1] - param[start + 13]);
    g3 = param[start + 2] + param[start + 8]*(x[2] - param[start + 14]);
    g4 = param[start + 3] + param[start + 9]*(x[3] - param[start + 15]);
    g5 = param[start + 4] + param[start + 10]*(x[4] - param[start + 16]);
    g6 = param[start + 5] + param[start + 11]*(x[5] - param[start + 17]);


    raug = param[start + 23];
    g1 += param[start + 18] + raug*c1;
    g2 += param[start + 19] + raug*c2;
    g3 += param[start + 20] + raug*c3;
    g4 += param[start + 21] + raug*c4;

    g5 += param[start + 18]*(-YffR) + param[start + 19]*(YffI) + param[start + 22]*(-x[5]) +
          raug*(-YffR)*c1 + raug*(YffI)*c2 + raug*(-x[5])*c5;
    g6 += param[start + 20]*(-YttR) + param[start + 21]*(YttI) + param[start + 22]*(-x[4]) +
          raug*(-YttR)*c3 + raug*(YttI)*c4 + raug*(-x[4])*c5;
    g7 = param[start + 18]*(-YftR) + param[start + 19]*(YftI) + param[start + 20]*(-YtfR) +
         param[start + 21]*(YtfI) + param[start + 22]*(2*x[6]) +
         raug*(-YftR)*c1 + raug*(YftI)*c2 + raug*(-YtfR)*c3 +
         raug*(YtfI)*c4 + raug*(2*x[6])*c5;
    g8 = param[start + 18]*(-YftI) + param[start + 19]*(-YftR) + param[start + 20]*(YtfI) +
         param[start + 21]*(YtfR) + param[start + 22]*(2*x[7]) +
         raug*(-YftI)*c1 + raug*(-YftR)*c2 + raug*(YtfI)*c3 +
         raug*(YtfR)*c4 + raug*(2*x[7])*c5;

    g[0] = g1;
    g[1] = g2;
    g[2] = g3;
    g[3] = g4;
    g[4] = g5;
    g[5] = g6;
    g[6] = g7;
    g[7] = g8;

    return;
}

void eval_h(int I, int n, double *x, double *A, double *param,
            double YffR, double YffI,
            double YftR, double YftI,
            double YttR, double YttI,
            double YtfR, double YtfI)
{
    double alrect, raug, c5;
    int start = 24*I;

    alrect = param[start + 22];
    raug = param[start + 23];
    c5 = (x[6]*x[6] + x[7]*x[7] - x[4]*x[5]);

    // 1st column
    A[0] = param[start + 6] + raug; // A[1,1]
    A[4] = raug*(-YffR); // A[5,1]
    A[6] = raug*(-YftR); // A[7,1]
    A[7] = raug*(-YftI); // A[8,1]

    // 2nd columns
    A[n + 1] = param[start + 7] + raug; // A[2,2]
    A[n + 4] = raug*(YffI);  // A[5,2]
    A[n + 6] = raug*(YftI);  // A[7,2]
    A[n + 7] = raug*(-YftR); // A[8,2]

    // 3rd column
    A[n*2 + 2] = param[start + 8] + raug; // A[3,3]
    A[n*2 + 5] = raug*(-YttR); // A[6,3]
    A[n*2 + 6] = raug*(-YtfR); // A[7,3]
    A[n*2 + 7] = raug*(YtfI);  // A[8,3]

    // 4th column
    A[n*3 + 3] = param[start + 9] + raug; // A[4,4]
    A[n*3 + 5] = raug*(YttI); // A[6,4]
    A[n*3 + 6] = raug*(YtfI); // A[7,4]
    A[n*3 + 7] = raug*(YtfR); // A[8,4]

    // 5th column
    A[n*4 + 4] = param[start + 10] + raug*(YffR*YffR) + raug*(YffI*YffI) + raug*(x[5]*x[5]); // A[5,5]
    A[n*4 + 5] = -(alrect + raug*c5) + raug*(x[4]*x[5]); // A[6,5]
    A[n*4 + 6] = raug*(YffR*YftR) + raug*(YffI*YftI) + raug*((-x[5])*(2*x[6])); // A[7,5]
    A[n*4 + 7] = raug*(YffR*YftI) + raug*(YffI*(-YftR)) + raug*((-x[5])*(2*x[7])); // A[8,5]

    // 6th column
    A[n*5 + 5] = param[start + 11] + raug*(YttR*YttR) + raug*(YttI*YttI) + raug*(x[4]*x[4]); // A[6,6]
    A[n*5 + 6] = raug*(YttR*YtfR) + raug*(YttI*YtfI) + raug*((-x[4])*(2*x[6])); // A[7,6]
    A[n*5 + 7] = raug*((-YttR)*YtfI) + raug*(YttI*YtfR) + raug*((-x[4])*(2*x[7])); // A[8,6]

    // 7th column
    A[n*6 + 6] = (alrect + raug*c5)*2 + raug*(YftR*YftR) + raug*(YftI*YftI) +
                 raug*(YtfR*YtfR) + raug*(YtfI*YtfI) + raug*((2*x[6])*(2*x[6])); // A[7,7]
    A[n*6 + 7] = raug*(YftR*YftI) + raug*(YftI*(-YftR)) + raug*((-YtfR)*YtfI) +
                 raug*(YtfI*YtfR) + raug*((2*x[6])*(2*x[7])); // A[8,7]

    // 8th column
    A[n*7 + 7] = (alrect + raug*c5)*2 + raug*(YftI*YftI) + raug*(YftR*YftR) +
                 raug*(YtfI*YtfI) + raug*(YtfR*YtfR) + raug*((2*x[7])*(2*x[7])); // A[8,8]

    for (int j = 0; j < n; j++) {
        for (int k = 0; k < j; k++) {
            A[n*j + k] = A[n*k + j];
        }
    }

    return;
}

void auglag(int nbranches, int n, int major_iter, int max_auglag,
            int pij_start, int qij_start,
            int pji_start, int qji_start,
            int wi_i_ij_start, int wi_j_ji_start,
            double mu_max,
            double *u_curr, double *v_curr, double *l_curr,
            double *rho, double *wRIij, double *param,
            double *_YffR, double *_YffI,
            double *_YftR, double *_YftI,
            double *_YttR, double *_YttI,
            double *_YtfR, double *_YtfI,
            double *frBound, double *toBound)
{
    bool terminate;
    int max_feval, max_minor, it, status, minor_iter;
    double mu, eta, omega;
    double YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI;

    double *x, *xl, *xu;

    x = (double *)calloc(n, sizeof(double));
    xl = (double *)calloc(n, sizeof(double));
    xu = (double *)calloc(n, sizeof(double));

    for (int I = 0; I < nbranches; I++) {
        for (int j = 0; j < n; j++) {
            xl[j] = -INF;
            xu[j] = INF;
        }

        xl[4] = frBound[2*I];
        xu[4] = frBound[2*I+1];
        xl[5] = toBound[2*I];
        xu[5] = toBound[2*I+1];

        x[0] = u_curr[pij_start+I];
        x[1] = u_curr[qij_start+I];
        x[2] = u_curr[pji_start+I];
        x[3] = u_curr[qji_start+I];
        x[4] = min(xu[4], max(xl[4], u_curr[wi_i_ij_start+I]));
        x[5] = min(xu[5], max(xl[5], u_curr[wi_j_ji_start+I]));
        x[6] = wRIij[2*I];
        x[7] = wRIij[2*I+1];

        YffR = _YffR[I]; YffI = _YffI[I];
        YftR = _YftR[I]; YftI = _YftI[I];
        YttR = _YttR[I]; YttI = _YttI[I];
        YtfR = _YtfR[I]; YtfI = _YtfI[I];

        int start = 24*I;
        param[start] = l_curr[pij_start+I];
        param[start + 1] = l_curr[qij_start+I];
        param[start + 2] = l_curr[pji_start+I];
        param[start + 3] = l_curr[qji_start+I];
        param[start + 4] = l_curr[wi_i_ij_start+I];
        param[start + 5] = l_curr[wi_j_ji_start+I];
        param[start + 6] = rho[pij_start+I];
        param[start + 7] = rho[qij_start+I];
        param[start + 8] = rho[pji_start+I];
        param[start + 9] = rho[qji_start+I];
        param[start + 10] = rho[wi_i_ij_start+I];
        param[start + 11] = rho[wi_j_ji_start+I];
        param[start + 12] = v_curr[pij_start+I];
        param[start + 13] = v_curr[qij_start+I];
        param[start + 14] = v_curr[pji_start+I];
        param[start + 15] = v_curr[qji_start+I];
        param[start + 16] = v_curr[wi_i_ij_start+I];
        param[start + 17] = v_curr[wi_j_ji_start+I];

        if (major_iter == 1) {
            param[start + 23] = 10.0;
            mu = 10.0;
        } else {
            mu = param[start + 23];
        }


        eta = 1 / pow(mu, 0.1);
        omega = 1 / mu;
        max_feval = 500;
        max_minor = 200;

        it = 0;
        terminate = false;

        double cviol1, cviol2, cviol3, cviol4, cviol5, cnorm;
        while (!terminate) {
            it += 1;

            // Solve the branch problem.

            driver_auglag(I, n, max_feval, max_minor, &status, &minor_iter,
                          x, xl, xu, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI,
                          &eval_f, &eval_grad_f, &eval_h);

            // Check the termination condition.
            cviol1 = x[0] - (YffR*x[4] + YftR*x[6] + YftI*x[7]);
            cviol2 = x[1] - (-YffI*x[4] - YftI*x[6] + YftR*x[7]);
            cviol3 = x[2] - (YttR*x[5] + YtfR*x[6] - YtfI*x[7]);
            cviol4 = x[3] - (-YttI*x[5] - YtfI*x[6] - YtfR*x[7]);
            cviol5 = x[6]*x[6] + x[7]*x[7] - x[4]*x[5];

            cnorm = max(abs(cviol1), max(abs(cviol2), max(abs(cviol3), max(abs(cviol4), abs(cviol5)))));

            if (cnorm <= eta) {
                if (cnorm <= 1e-6) {
                    terminate = true;
                } else {
                    param[start + 18] += mu*cviol1;
                    param[start + 19] += mu*cviol2;
                    param[start + 20] += mu*cviol3;
                    param[start + 21] += mu*cviol4;
                    param[start + 22] += mu*cviol5;

                    eta = eta / pow(mu, 0.9);
                    omega  = omega / mu;
                }
            } else {
                mu = min(mu_max, mu*10);
                eta = 1 / pow(mu, 0.1);
                omega = 1 / mu;
                param[start + 23] = mu;
            }

            if (it >= max_auglag) {
                terminate = true;
            }
        }

        u_curr[pij_start+I] = x[0];
        u_curr[qij_start+I] = x[1];
        u_curr[pji_start+I] = x[2];
        u_curr[qji_start+I] = x[3];
        u_curr[wi_i_ij_start+I] = x[4];
        u_curr[wi_j_ji_start+I] = x[5];
        wRIij[2*I] = x[6];
        wRIij[2*I+1] = x[7];
        param[start + 23] = mu;
    }

    free(x);
    free(xl);
    free(xu);

    return;
}