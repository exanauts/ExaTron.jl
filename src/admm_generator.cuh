__global__
void update_generator_kernel(double baseMVA, int ngens, int pg_start, int qg_start,
                             double *u, double *v, double *l, double *rho,
                             double *pgmin, double *pgmax, double *qgmin, double *qgmax,
                             double *c2, double *c1, double *c0)
{
    int I = threadIdx.x + (blockDim.x * blockIdx.x);

    if (I < ngens) {
        u[pg_start+I] = max(pgmin[I],
                              min(pgmax[I],
                            (-(c1[I]*baseMVA + l[pg_start+I] -
                               rho[pg_start+I]*v[pg_start+I])) / (2*c2[I]*(baseMVA*baseMVA) + rho[pg_start+I])));
        u[qg_start+I] = max(qgmin[I],
                              min(qgmax[I],
                            (-(l[qg_start+I] - rho[qg_start+I]*v[qg_start+I])) / rho[qg_start+I]));
    }
    return;
}

void update_generator(double baseMVA, int ngens, int pg_start, int qg_start,
                      double *u, double *v, double *l, double *rho,
                      double *pgmin, double *pgmax, double *qgmin, double *qgmax,
                      double *c2, double *c1, double *c0)
{
    for (int I = 0; I < ngens; I++) {
        u[pg_start+I] = max(pgmin[I],
                            min(pgmax[I],
                          (-(c1[I]*baseMVA + l[pg_start+I] -
                              rho[pg_start+I]*v[pg_start+I])) / (2*c2[I]*(baseMVA*baseMVA) + rho[pg_start+I])));
        u[qg_start+I] = max(qgmin[I],
                            min(qgmax[I],
                          (-(l[qg_start+I] - rho[qg_start+I]*v[qg_start+I])) / rho[qg_start+I]));
    }
    return;
}

