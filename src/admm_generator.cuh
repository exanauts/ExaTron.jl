__global__
void update_generator_kernel(double baseMVA, int ngens, int generator_start,
                             double *u, double *v, double *l, double *rho,
                             double *pgmin, double *pgmax, double *qgmin, double *qgmax,
                             double *c2, double *c1, double *c0)
{
    int I = threadIdx.x + (blockDim.x * blockIdx.x);

    if (I < ngens) {
        int pg_idx = generator_start + 2*I;
        int qg_idx = generator_start + 2*I + 1;
        u[pg_idx] = max(pgmin[I],
                        min(pgmax[I],
                            (-(c1[I]*baseMVA + l[pg_idx] -
                               rho[pg_idx]*v[pg_idx])) / (2*c2[I]*(baseMVA*baseMVA) + rho[pg_idx])));
        u[qg_idx] = max(qgmin[I],
                        min(qgmax[I],
                            (-(l[qg_idx] - rho[qg_idx]*v[qg_idx])) / rho[qg_idx]));
    }
    return;
}

void update_generator(double baseMVA, int ngens, int generator_start,
                      double *u, double *v, double *l, double *rho,
                      double *pgmin, double *pgmax, double *qgmin, double *qgmax,
                      double *c2, double *c1, double *c0)
{
    for (int I = 0; I < ngens; I++) {
        int pg_idx = generator_start + 2*I;
        int qg_idx = generator_start + 2*I + 1;
        u[pg_idx] = max(pgmin[I],
                        min(pgmax[I],
                            (-(c1[I]*baseMVA + l[pg_idx] -
                              rho[pg_idx]*v[pg_idx])) / (2*c2[I]*(baseMVA*baseMVA) + rho[pg_idx])));
        u[qg_idx] = max(qgmin[I],
                        min(qgmax[I],
                            (-(l[qg_idx] - rho[qg_idx]*v[qg_idx])) / rho[qg_idx]));
    }
    return;
}

