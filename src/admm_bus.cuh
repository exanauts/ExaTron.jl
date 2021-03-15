__global__
void update_bus_kernel(double baseMVA, int nbuses, int pg_start, int qg_start, int pij_start, int qij_start,
                       int pji_start, int qji_start, int wi_i_ij_start, int wi_j_ji_start, int ti_i_ij_start, int ti_j_ji_start,
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
            common_wi += l[wi_i_ij_start + fr_idx[k]] + rho[wi_i_ij_start + fr_idx[k]]*u[wi_i_ij_start + fr_idx[k]];
            common_ti += l[ti_i_ij_start + fr_idx[k]] + rho[ti_i_ij_start + fr_idx[k]]*u[ti_i_ij_start + fr_idx[k]];
            inv_rhosum_pij_ji += 1.0 / rho[pij_start + fr_idx[k]];
            inv_rhosum_qij_ji += 1.0 / rho[qij_start + fr_idx[k]];
            rhosum_wi_ij_ji += rho[wi_i_ij_start + fr_idx[k]];
            rhosum_ti_ij_ji += rho[ti_i_ij_start + fr_idx[k]];
        }
        for (int k = to_start[I]; k < to_start[I+1]; k++) {
            common_wi += l[wi_j_ji_start + to_idx[k]] + rho[wi_j_ji_start + to_idx[k]]*u[wi_j_ji_start + to_idx[k]];
            common_ti += l[ti_j_ji_start + to_idx[k]] + rho[ti_j_ji_start + to_idx[k]]*u[ti_j_ji_start + to_idx[k]];
            inv_rhosum_pij_ji += 1.0 / rho[pji_start + to_idx[k]];
            inv_rhosum_qij_ji += 1.0 / rho[qji_start + to_idx[k]];
            rhosum_wi_ij_ji += rho[wi_j_ji_start + to_idx[k]];
            rhosum_ti_ij_ji += rho[ti_j_ji_start + to_idx[k]];
        }

        common_wi /= rhosum_wi_ij_ji;

        double rhs1 = 0;
        double rhs2 = 0;
        double inv_rhosum_pg = 0;
        double inv_rhosum_qg = 0;

        for (int g = gen_start[I]; g < gen_start[I+1]; g++) {
            rhs1 += u[pg_start + gen_idx[g]] + (l[pg_start + gen_idx[g]]/rho[pg_start + gen_idx[g]]);
            rhs2 += u[qg_start + gen_idx[g]] + (l[qg_start + gen_idx[g]]/rho[qg_start + gen_idx[g]]);
            inv_rhosum_pg += 1.0 / rho[pg_start + gen_idx[g]];
            inv_rhosum_qg += 1.0 / rho[qg_start + gen_idx[g]];
        }

        rhs1 -= (pd[I] / baseMVA);
        rhs2 -= (qd[I] / baseMVA);

        for (int k = fr_start[I]; k < fr_start[I+1]; k++) {
            rhs1 -= u[pij_start + fr_idx[k]] + (l[pij_start + fr_idx[k]]/rho[pij_start + fr_idx[k]]);
            rhs2 -= u[qij_start + fr_idx[k]] + (l[qij_start + fr_idx[k]]/rho[qij_start + fr_idx[k]]);
        }

        for (int k = to_start[I]; k < to_start[I+1]; k++) {
            rhs1 -= u[pji_start + to_idx[k]] + (l[pji_start + to_idx[k]]/rho[pji_start + to_idx[k]]);
            rhs2 -= u[qji_start + to_idx[k]] + (l[qji_start + to_idx[k]]/rho[qji_start + to_idx[k]]);
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
            int g = gen_idx[k];
            v[pg_start + g] = u[pg_start + g] + (l[pg_start + g] - mu1) / rho[pg_start + g];
            v[qg_start + g] = u[qg_start + g] + (l[qg_start + g] - mu2) / rho[qg_start + g];
        }
        for (int j = fr_start[I]; j < fr_start[I+1]; j++) {
            int k = fr_idx[j];
            v[pij_start + k] = u[pij_start + k] + (l[pij_start + k] + mu1) / rho[pij_start + k];
            v[qij_start + k] = u[qij_start + k] + (l[qij_start + k] + mu2) / rho[qij_start + k];
            v[wi_i_ij_start + k] = wi;
            v[ti_i_ij_start + k] = ti;
        }
        for (int j = to_start[I]; j < to_start[I+1]; j++) {
            int k = to_idx[j];
            v[pji_start + k] = u[pji_start + k] + (l[pji_start + k] + mu1) / rho[pji_start + k];
            v[qji_start + k] = u[qji_start + k] + (l[qji_start + k] + mu2) / rho[qji_start + k];
            v[wi_j_ji_start + k] = wi;
            v[ti_j_ji_start + k] = ti;
        }
    }

    return;
}

void update_bus(double baseMVA, int nbuses, int pg_start, int qg_start, int pij_start, int qij_start,
                int pji_start, int qji_start, int wi_i_ij_start, int wi_j_ji_start, int ti_i_ij_start, int ti_j_ji_start,
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
            common_wi += l[wi_i_ij_start + fr_idx[k]] + rho[wi_i_ij_start + fr_idx[k]]*u[wi_i_ij_start + fr_idx[k]];
            common_ti += l[ti_i_ij_start + fr_idx[k]] + rho[ti_i_ij_start + fr_idx[k]]*u[ti_i_ij_start + fr_idx[k]];
            inv_rhosum_pij_ji += 1.0 / rho[pij_start + fr_idx[k]];
            inv_rhosum_qij_ji += 1.0 / rho[qij_start + fr_idx[k]];
            rhosum_wi_ij_ji += rho[wi_i_ij_start + fr_idx[k]];
            rhosum_ti_ij_ji += rho[ti_i_ij_start + fr_idx[k]];
        }
        for (int k = to_start[I]; k < to_start[I+1]; k++) {
            common_wi += l[wi_j_ji_start + to_idx[k]] + rho[wi_j_ji_start + to_idx[k]]*u[wi_j_ji_start + to_idx[k]];
            common_ti += l[ti_j_ji_start + to_idx[k]] + rho[ti_j_ji_start + to_idx[k]]*u[ti_j_ji_start + to_idx[k]];
            inv_rhosum_pij_ji += 1.0 / rho[pji_start + to_idx[k]];
            inv_rhosum_qij_ji += 1.0 / rho[qji_start + to_idx[k]];
            rhosum_wi_ij_ji += rho[wi_j_ji_start + to_idx[k]];
            rhosum_ti_ij_ji += rho[ti_j_ji_start + to_idx[k]];
        }

        common_wi /= rhosum_wi_ij_ji;

        double rhs1 = 0;
        double rhs2 = 0;
        double inv_rhosum_pg = 0;
        double inv_rhosum_qg = 0;

        for (int g = gen_start[I]; g < gen_start[I+1]; g++) {
            rhs1 += u[pg_start + gen_idx[g]] + (l[pg_start + gen_idx[g]]/rho[pg_start + gen_idx[g]]);
            rhs2 += u[qg_start + gen_idx[g]] + (l[qg_start + gen_idx[g]]/rho[qg_start + gen_idx[g]]);
            inv_rhosum_pg += 1.0 / rho[pg_start + gen_idx[g]];
            inv_rhosum_qg += 1.0 / rho[qg_start + gen_idx[g]];
        }

        rhs1 -= (pd[I] / baseMVA);
        rhs2 -= (qd[I] / baseMVA);

        for (int k = fr_start[I]; k < fr_start[I+1]; k++) {
            rhs1 -= u[pij_start + fr_idx[k]] + (l[pij_start + fr_idx[k]]/rho[pij_start + fr_idx[k]]);
            rhs2 -= u[qij_start + fr_idx[k]] + (l[qij_start + fr_idx[k]]/rho[qij_start + fr_idx[k]]);
        }

        for (int k = to_start[I]; k < to_start[I+1]; k++) {
            rhs1 -= u[pji_start + to_idx[k]] + (l[pji_start + to_idx[k]]/rho[pji_start + to_idx[k]]);
            rhs2 -= u[qji_start + to_idx[k]] + (l[qji_start + to_idx[k]]/rho[qji_start + to_idx[k]]);
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
            int g = gen_idx[k];
            v[pg_start + g] = u[pg_start + g] + (l[pg_start + g] - mu1) / rho[pg_start + g];
            v[qg_start + g] = u[qg_start + g] + (l[qg_start + g] - mu2) / rho[qg_start + g];
        }
        for (int j = fr_start[I]; j < fr_start[I+1]; j++) {
            int k = fr_idx[j];
            v[pij_start + k] = u[pij_start + k] + (l[pij_start + k] + mu1) / rho[pij_start + k];
            v[qij_start + k] = u[qij_start + k] + (l[qij_start + k] + mu2) / rho[qij_start + k];
            v[wi_i_ij_start + k] = wi;
            v[ti_i_ij_start + k] = ti;
        }
        for (int j = to_start[I]; j < to_start[I+1]; j++) {
            int k = to_idx[j];
            v[pji_start + k] = u[pji_start + k] + (l[pji_start + k] + mu1) / rho[pji_start + k];
            v[qji_start + k] = u[qji_start + k] + (l[qji_start + k] + mu2) / rho[qji_start + k];
            v[wi_j_ji_start + k] = wi;
            v[ti_j_ji_start + k] = ti;
        }
    }
    return;
}