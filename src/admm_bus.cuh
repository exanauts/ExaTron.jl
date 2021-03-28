__global__
void update_bus_kernel(double baseMVA, int nbuses, int generator_start, int line_start,
                       int *fr_start, int *fr_idx, int *to_start, int *to_idx, int *gen_start, int *gen_idx,
                       double *pd, double *qd, double *u, double *v, double *l, double *rho, double *YshR, double *YshI)
{
    int I = threadIdx.x + (blockDim.x * blockIdx.x);
    if (I < nbuses) {
        double common_wi = 0;
        double common_ti = 0;
        double inv_rhosum_pij_ji = 0;
        double inv_rhosum_qij_ji = 0;
        double rhosum_wi_ij_ji = 0;
        double rhosum_ti_ij_ji = 0;

        for (int k = fr_start[I]; k < fr_start[I+1]; k++) {
            int pij_idx = line_start + 8*fr_idx[k];
            common_wi += l[pij_idx+4] + rho[pij_idx+4]*u[pij_idx+4];
            common_ti += l[pij_idx+6] + rho[pij_idx+6]*u[pij_idx+6];
            inv_rhosum_pij_ji += 1.0 / rho[pij_idx];
            inv_rhosum_qij_ji += 1.0 / rho[pij_idx+1];
            rhosum_wi_ij_ji += rho[pij_idx+4];
            rhosum_ti_ij_ji += rho[pij_idx+6];
        }
        for (int k = to_start[I]; k < to_start[I+1]; k++) {
            int pij_idx = line_start + 8*to_idx[k];
            common_wi += l[pij_idx+5] + rho[pij_idx+5]*u[pij_idx+5];
            common_ti += l[pij_idx+7] + rho[pij_idx+7]*u[pij_idx+7];
            inv_rhosum_pij_ji += 1.0 / rho[pij_idx+2];
            inv_rhosum_qij_ji += 1.0 / rho[pij_idx+3];
            rhosum_wi_ij_ji += rho[pij_idx+5];
            rhosum_ti_ij_ji += rho[pij_idx+7];
        }

        common_wi /= rhosum_wi_ij_ji;

        double rhs1 = 0;
        double rhs2 = 0;
        double inv_rhosum_pg = 0;
        double inv_rhosum_qg = 0;

        for (int g = gen_start[I]; g < gen_start[I+1]; g++) {
            int pg_idx = generator_start + 2*gen_idx[g];
            rhs1 += u[pg_idx] + (l[pg_idx]/rho[pg_idx]);
            rhs2 += u[pg_idx+1] + (l[pg_idx+1]/rho[pg_idx+1]);
            inv_rhosum_pg += 1.0 / rho[pg_idx];
            inv_rhosum_qg += 1.0 / rho[pg_idx+1];
        }

        rhs1 -= (pd[I] / baseMVA);
        rhs2 -= (qd[I] / baseMVA);

        for (int k = fr_start[I]; k < fr_start[I+1]; k++) {
            int pij_idx = line_start + 8*fr_idx[k];
            rhs1 -= u[pij_idx] + (l[pij_idx]/rho[pij_idx]);
            rhs2 -= u[pij_idx+1] + (l[pij_idx+1]/rho[pij_idx+1]);
        }

        for (int k = to_start[I]; k < to_start[I+1]; k++) {
            int pij_idx = line_start + 8*to_idx[k];
            rhs1 -= u[pij_idx+2] + (l[pij_idx+2]/rho[pij_idx+2]);
            rhs2 -= u[pij_idx+3] + (l[pij_idx+3]/rho[pij_idx+3]);
        }

        rhs1 -= YshR[I]*common_wi;
        rhs2 += YshI[I]*common_wi;

        double A11 = (inv_rhosum_pg + inv_rhosum_pij_ji) + ((YshR[I]*YshR[I]) / rhosum_wi_ij_ji);
        double A12 = -YshR[I]*(YshI[I] / rhosum_wi_ij_ji);
        double A21 = A12;
        double A22 = (inv_rhosum_qg + inv_rhosum_qij_ji) + ((YshI[I]*YshI[I]) / rhosum_wi_ij_ji);
        double mu2 = (rhs2 - (A21/A11)*rhs1) / (A22 - (A21/A11)*A12);
        double mu1 = (rhs1 - A12*mu2) / A11;
        //mu = A \ [rhs1 ; rhs2]
        double wi = common_wi + ( (YshR[I]*mu1 - YshI[I]*mu2) / rhosum_wi_ij_ji );
        double ti = common_ti / rhosum_ti_ij_ji;

        for (int k = gen_start[I]; k < gen_start[I+1]; k++) {
            int pg_idx = generator_start + 2*gen_idx[k];
            v[pg_idx] = u[pg_idx] + (l[pg_idx] - mu1) / rho[pg_idx];
            v[pg_idx+1] = u[pg_idx+1] + (l[pg_idx+1] - mu2) / rho[pg_idx+1];
        }
        for (int j = fr_start[I]; j < fr_start[I+1]; j++) {
            int pij_idx = line_start + 8*fr_idx[j];
            v[pij_idx] = u[pij_idx] + (l[pij_idx] + mu1) / rho[pij_idx];
            v[pij_idx+1] = u[pij_idx+1] + (l[pij_idx+1] + mu2) / rho[pij_idx+1];
            v[pij_idx+4] = wi;
            v[pij_idx+6] = ti;
        }
        for (int j = to_start[I]; j < to_start[I+1]; j++) {
            int pij_idx = line_start + 8*to_idx[j];
            v[pij_idx+2] = u[pij_idx+2] + (l[pij_idx+2] + mu1) / rho[pij_idx+2];
            v[pij_idx+3] = u[pij_idx+3] + (l[pij_idx+3] + mu2) / rho[pij_idx+3];
            v[pij_idx+5] = wi;
            v[pij_idx+7] = ti;
        }
    }

    return;
}

void update_bus(double baseMVA, int nbuses, int generator_start, int line_start,
                int *fr_start, int *fr_idx, int *to_start, int *to_idx, int *gen_start, int *gen_idx,
                double *pd, double *qd, double *u, double *v, double *l, double *rho, double *YshR, double *YshI)
{
    for (int I = 0; I < nbuses; I++) {
        double common_wi = 0;
        double common_ti = 0;
        double inv_rhosum_pij_ji = 0;
        double inv_rhosum_qij_ji = 0;
        double rhosum_wi_ij_ji = 0;
        double rhosum_ti_ij_ji = 0;

        for (int k = fr_start[I]; k < fr_start[I+1]; k++) {
            int pij_idx = line_start + 8*fr_idx[k];
            common_wi += l[pij_idx+4] + rho[pij_idx+4]*u[pij_idx+4];
            common_ti += l[pij_idx+6] + rho[pij_idx+6]*u[pij_idx+6];
            inv_rhosum_pij_ji += 1.0 / rho[pij_idx];
            inv_rhosum_qij_ji += 1.0 / rho[pij_idx+1];
            rhosum_wi_ij_ji += rho[pij_idx+4];
            rhosum_ti_ij_ji += rho[pij_idx+6];
        }
        for (int k = to_start[I]; k < to_start[I+1]; k++) {
            int pij_idx = line_start + 8*to_idx[k];
            common_wi += l[pij_idx+5] + rho[pij_idx+5]*u[pij_idx+5];
            common_ti += l[pij_idx+7] + rho[pij_idx+7]*u[pij_idx+7];
            inv_rhosum_pij_ji += 1.0 / rho[pij_idx+2];
            inv_rhosum_qij_ji += 1.0 / rho[pij_idx+3];
            rhosum_wi_ij_ji += rho[pij_idx+5];
            rhosum_ti_ij_ji += rho[pij_idx+7];
        }

        common_wi /= rhosum_wi_ij_ji;

        double rhs1 = 0;
        double rhs2 = 0;
        double inv_rhosum_pg = 0;
        double inv_rhosum_qg = 0;

        for (int g = gen_start[I]; g < gen_start[I+1]; g++) {
            int pg_idx = generator_start + 2*gen_idx[g];
            rhs1 += u[pg_idx] + (l[pg_idx]/rho[pg_idx]);
            rhs2 += u[pg_idx+1] + (l[pg_idx+1]/rho[pg_idx+1]);
            inv_rhosum_pg += 1.0 / rho[pg_idx];
            inv_rhosum_qg += 1.0 / rho[pg_idx+1];
        }

        rhs1 -= (pd[I] / baseMVA);
        rhs2 -= (qd[I] / baseMVA);

        for (int k = fr_start[I]; k < fr_start[I+1]; k++) {
            int pij_idx = line_start + 8*fr_idx[k];
            rhs1 -= u[pij_idx] + (l[pij_idx]/rho[pij_idx]);
            rhs2 -= u[pij_idx+1] + (l[pij_idx+1]/rho[pij_idx+1]);
        }

        for (int k = to_start[I]; k < to_start[I+1]; k++) {
            int pij_idx = line_start + 8*to_idx[k];
            rhs1 -= u[pij_idx+2] + (l[pij_idx+2]/rho[pij_idx+2]);
            rhs2 -= u[pij_idx+3] + (l[pij_idx+3]/rho[pij_idx+3]);
        }

        rhs1 -= YshR[I]*common_wi;
        rhs2 += YshI[I]*common_wi;

        double A11 = (inv_rhosum_pg + inv_rhosum_pij_ji) + ((YshR[I]*YshR[I]) / rhosum_wi_ij_ji);
        double A12 = -YshR[I]*(YshI[I] / rhosum_wi_ij_ji);
        double A21 = A12;
        double A22 = (inv_rhosum_qg + inv_rhosum_qij_ji) + ((YshI[I]*YshI[I]) / rhosum_wi_ij_ji);
        double mu2 = (rhs2 - (A21/A11)*rhs1) / (A22 - (A21/A11)*A12);
        double mu1 = (rhs1 - A12*mu2) / A11;
        //mu = A \ [rhs1 ; rhs2]
        double wi = common_wi + ( (YshR[I]*mu1 - YshI[I]*mu2) / rhosum_wi_ij_ji );
        double ti = common_ti / rhosum_ti_ij_ji;

        for (int k = gen_start[I]; k < gen_start[I+1]; k++) {
            int pg_idx = generator_start + 2*gen_idx[k];
            v[pg_idx] = u[pg_idx] + (l[pg_idx] - mu1) / rho[pg_idx];
            v[pg_idx+1] = u[pg_idx+1] + (l[pg_idx+1] - mu2) / rho[pg_idx+1];
        }
        for (int j = fr_start[I]; j < fr_start[I+1]; j++) {
            int pij_idx = line_start + 8*fr_idx[j];
            v[pij_idx] = u[pij_idx] + (l[pij_idx] + mu1) / rho[pij_idx];
            v[pij_idx+1] = u[pij_idx+1] + (l[pij_idx+1] + mu2) / rho[pij_idx+1];
            v[pij_idx+4] = wi;
            v[pij_idx+6] = ti;
        }
        for (int j = to_start[I]; j < to_start[I+1]; j++) {
            int pij_idx = line_start + 8*to_idx[j];
            v[pij_idx+2] = u[pij_idx+2] + (l[pij_idx+2] + mu1) / rho[pij_idx+2];
            v[pij_idx+3] = u[pij_idx+3] + (l[pij_idx+3] + mu2) / rho[pij_idx+3];
            v[pij_idx+5] = wi;
            v[pij_idx+7] = ti;
        }
    }
    return;
}