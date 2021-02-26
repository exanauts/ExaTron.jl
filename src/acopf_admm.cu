#include <iostream>
#include <random>
#include <math.h>
#include <assert.h>

#include "consts.cuh"
#include "network.cuh"
#include "print.cuh"
#include "admittance.cuh"
#include "parse_mat.cuh"

#include "creorder.cuh"
#include "csol.cuh"
#include "cssyax.cuh"
#include "caxpy.cuh"
#include "ccopy.cuh"
#include "cmid.cuh"
#include "cscal.cuh"
#include "cdot.cuh"
#include "ctrqsol.cuh"
#include "cnrm2.cuh"
#include "cgpnorm.cuh"
#include "cgpstep.cuh"
#include "cbreakpt.cuh"
#include "cprsrch.cuh"
#include "ccauchy.cuh"
#include "ctrpcg.cuh"
#include "cicf.cuh"
#include "cicfs.cuh"
#include "cspcg.cuh"
#include "ctron.cuh"
#include "cdriver.cuh"

#include "gputest_utilities.cuh"

#include "admm_auglag.cuh"
#include "admm_generator.cuh"
#include "admm_bus.cuh"

static void usage(const char *progname)
{
    printf("Usage: %s file iterlim rho_pq rho_va use_gpu\n", progname);
    printf("  file   : MATPOWER power network file\n");
    printf("  iterlim: iteration limit of ADMM\n");
    printf("  rho_pq : initial rho for power flows\n");
    printf("  rho_va : initial rho for voltages\n");
    printf("  use_gpu: 0 for CPU and 1 for GPU\n");
}

void get_generator_data(Network *nw, double **_pgmin, double **_pgmax,
                        double **_qgmin, double **_qgmax, double **_c2,
                        double **_c1, double **_c0, bool gpu = false)
{
    int memSize;
    double *pgmin, *pgmax, *qgmin, *qgmax, *c2, *c1, *c0;

    memSize = sizeof(double)*nw->active_ngen;
    pgmin = (double *)malloc(memSize);
    pgmax = (double *)malloc(memSize);
    qgmin = (double *)malloc(memSize);
    qgmax = (double *)malloc(memSize);
    c2 = (double *)malloc(memSize);
    c1 = (double *)malloc(memSize);
    c0 = (double *)malloc(memSize);

    for (int i = 0; i < nw->active_ngen; i++) {
        pgmin[i] = nw->gen[i].pb / nw->baseMVA;
        pgmax[i] = nw->gen[i].pt / nw->baseMVA;
        qgmin[i] = nw->gen[i].qb / nw->baseMVA;
        qgmax[i] = nw->gen[i].qt / nw->baseMVA;
        c2[i] = nw->objcost[nw->gen[i].objcost_start];
        c1[i] = nw->objcost[nw->gen[i].objcost_start + 1];
        c0[i] = nw->objcost[nw->gen[i].objcost_start + 2];
    }

    if (gpu) {
        double *cu_pgmin, *cu_pgmax, *cu_qgmin, *cu_qgmax, *cu_c2, *cu_c1, *cu_c0;

        cudaMalloc(&cu_pgmin, memSize);
        cudaMalloc(&cu_pgmax, memSize);
        cudaMalloc(&cu_qgmin, memSize);
        cudaMalloc(&cu_qgmax, memSize);
        cudaMalloc(&cu_c2, memSize);
        cudaMalloc(&cu_c1, memSize);
        cudaMalloc(&cu_c0, memSize);
        cudaMemcpy(cu_pgmin, pgmin, memSize, cudaMemcpyHostToDevice);
        cudaMemcpy(cu_pgmax, pgmax, memSize, cudaMemcpyHostToDevice);
        cudaMemcpy(cu_qgmin, qgmin, memSize, cudaMemcpyHostToDevice);
        cudaMemcpy(cu_qgmax, qgmax, memSize, cudaMemcpyHostToDevice);
        cudaMemcpy(cu_c2, c2, memSize, cudaMemcpyHostToDevice);
        cudaMemcpy(cu_c1, c1, memSize, cudaMemcpyHostToDevice);
        cudaMemcpy(cu_c0, c0, memSize, cudaMemcpyHostToDevice);

        (*_pgmin) = cu_pgmin;
        (*_pgmax) = cu_pgmax;
        (*_qgmin) = cu_qgmin;
        (*_qgmax) = cu_qgmax;
        (*_c2) = cu_c2;
        (*_c1) = cu_c1;
        (*_c0) = cu_c0;
    } else {
        (*_pgmin) = pgmin;
        (*_pgmax) = pgmax;
        (*_qgmin) = qgmin;
        (*_qgmax) = qgmax;
        (*_c2) = c2;
        (*_c1) = c1;
        (*_c0) = c0;
    }

    return;
}

void get_bus_data(Network *nw, int **_fr_idx, int **_to_idx, int **_gen_idx,
                  int **_fr_start, int **_to_start, int **_gen_start,
                  double **_pd, double **_qd, bool gpu = false)
{
    int *fr_idx, *to_idx, *gen_idx, *fr_start, *to_start, *gen_start;
    double *pd, *qd;

    fr_idx = (int *)malloc(sizeof(int)*nw->active_nbranch);
    to_idx = (int *)malloc(sizeof(int)*nw->active_nbranch);
    gen_idx = (int *)malloc(sizeof(int)*nw->active_ngen);
    fr_start = (int *)malloc(sizeof(int)*(nw->nbus+1));
    to_start = (int *)malloc(sizeof(int)*(nw->nbus+1));
    gen_start = (int *)malloc(sizeof(int)*(nw->nbus+1));
    pd = (double *)malloc(sizeof(double)*nw->nbus);
    qd = (double *)malloc(sizeof(double)*nw->nbus);

    int l = 0;
    for (int i = 0; i < nw->nbus; i++) {
        fr_start[i] = l;
        for (int k = 0; k < nw->nfrom[i]; k++) {
            fr_idx[l] = *(nw->frombus[i] + k);
            l++;
        }
    }
    fr_start[nw->nbus] = l;

    l = 0;
    for (int i = 0; i < nw->nbus; i++) {
        to_start[i] = l;
        for (int k = 0; k < nw->nto[i]; k++) {
            to_idx[l] = *(nw->tobus[i] + k);
            l++;
        }
    }
    to_start[nw->nbus] = l;

    l = 0;
    for (int i = 0; i < nw->nbus; i++) {
        gen_start[i] = l;
        for (int k = 0; k < nw->ngenbus[i]; k++) {
            gen_idx[l] = *(nw->genbus[i] + k);
            l++;
        }
    }
    gen_start[nw->nbus] = l;

    for (int i = 0; i < nw->nbus; i++) {
        pd[i] = nw->bus[i].pd;
        qd[i] = nw->bus[i].qd;
    }

    if (gpu) {
        int *cu_fridx, *cu_toidx, *cu_genidx, *cu_frstart, *cu_tostart, *cu_genstart;
        double *cu_pd, *cu_qd;

        cudaMalloc(&cu_fridx, sizeof(int)*nw->active_nbranch);
        cudaMalloc(&cu_toidx, sizeof(int)*nw->active_nbranch);
        cudaMalloc(&cu_genidx, sizeof(int)*nw->active_ngen);
        cudaMalloc(&cu_frstart, sizeof(int)*(nw->nbus+1));
        cudaMalloc(&cu_tostart, sizeof(int)*(nw->nbus+1));
        cudaMalloc(&cu_genstart, sizeof(int)*(nw->nbus+1));
        cudaMalloc(&cu_pd, sizeof(double)*nw->nbus);
        cudaMalloc(&cu_qd, sizeof(double)*nw->nbus);
        cudaMemcpy(cu_fridx, fr_idx, sizeof(int)*nw->active_nbranch, cudaMemcpyHostToDevice);
        cudaMemcpy(cu_toidx, to_idx, sizeof(int)*nw->active_nbranch, cudaMemcpyHostToDevice);
        cudaMemcpy(cu_genidx, gen_idx, sizeof(int)*nw->active_ngen, cudaMemcpyHostToDevice);
        cudaMemcpy(cu_frstart, fr_start, sizeof(int)*(nw->nbus+1), cudaMemcpyHostToDevice);
        cudaMemcpy(cu_tostart, to_start, sizeof(int)*(nw->nbus+1), cudaMemcpyHostToDevice);
        cudaMemcpy(cu_genstart, gen_start, sizeof(int)*(nw->nbus+1), cudaMemcpyHostToDevice);
        cudaMemcpy(cu_pd, pd, sizeof(double)*nw->nbus, cudaMemcpyHostToDevice);
        cudaMemcpy(cu_qd, qd, sizeof(double)*nw->nbus, cudaMemcpyHostToDevice);

        (*_fr_idx) = cu_fridx;
        (*_to_idx) = cu_toidx;
        (*_gen_idx) = cu_genidx;
        (*_fr_start) = cu_frstart;
        (*_to_start) = cu_tostart;
        (*_gen_start) = cu_genstart;
        (*_pd) = cu_pd;
        (*_qd) = cu_qd;
    } else {
        (*_fr_idx) = fr_idx;
        (*_to_idx) = to_idx;
        (*_gen_idx) = gen_idx;
        (*_fr_start) = fr_start;
        (*_to_start) = to_start;
        (*_gen_start) = gen_start;
        (*_pd) = pd;
        (*_qd) = qd;
    }

    return;
}

void get_branch_data(Network *nw, double **_YshR, double **_YshI,
                     double **_YffR, double **_YffI, double **_YftR, double **_YftI,
                     double **_YttR, double **_YttI, double **_YtfR, double **_YtfI,
                     double **_frbound, double **_tobound,
                     bool gpu = false)
{
    double *YshR, *YshI, *YffR, *YffI, *YftR, *YftI, *YttR, *YttI, *YtfR, *YtfI;
    double *frbound, *tobound;

    YshR = (double *)malloc(sizeof(double)*nw->nbus);
    YshI = (double *)malloc(sizeof(double)*nw->nbus);
    YffR = (double *)malloc(sizeof(double)*nw->active_nbranch);
    YffI = (double *)malloc(sizeof(double)*nw->active_nbranch);
    YftR = (double *)malloc(sizeof(double)*nw->active_nbranch);
    YftI = (double *)malloc(sizeof(double)*nw->active_nbranch);
    YttR = (double *)malloc(sizeof(double)*nw->active_nbranch);
    YttI = (double *)malloc(sizeof(double)*nw->active_nbranch);
    YtfR = (double *)malloc(sizeof(double)*nw->active_nbranch);
    YtfI = (double *)malloc(sizeof(double)*nw->active_nbranch);
    frbound = (double *)malloc(sizeof(double)*(2*nw->active_nbranch));
    tobound = (double *)malloc(sizeof(double)*(2*nw->active_nbranch));

    for (int i = 0; i < nw->nbus; i++) {
        YshR[i] = nw->bus[i].gs / nw->baseMVA;
        YshI[i] = nw->bus[i].bs / nw->baseMVA;
    }

    memcpy(YffR, nw->YffR, sizeof(double)*nw->active_nbranch);
    memcpy(YffI, nw->YffI, sizeof(double)*nw->active_nbranch);
    memcpy(YftR, nw->YftR, sizeof(double)*nw->active_nbranch);
    memcpy(YftI, nw->YftI, sizeof(double)*nw->active_nbranch);
    memcpy(YttR, nw->YttR, sizeof(double)*nw->active_nbranch);
    memcpy(YttI, nw->YttI, sizeof(double)*nw->active_nbranch);
    memcpy(YtfR, nw->YtfR, sizeof(double)*nw->active_nbranch);
    memcpy(YtfI, nw->YtfI, sizeof(double)*nw->active_nbranch);

    for (int i = 0; i < nw->active_nbranch; i++) {
        int fr = nw->b2i[nw->branch[i].fr];
        int to = nw->b2i[nw->branch[i].to];
        frbound[2*i] = nw->bus[fr].nvlo * nw->bus[fr].nvlo;
        frbound[2*i+1] = nw->bus[fr].nvhi * nw->bus[fr].nvhi;
        tobound[2*i] = nw->bus[to].nvlo * nw->bus[to].nvlo;
        tobound[2*i+1] = nw->bus[to].nvhi * nw->bus[to].nvhi;
    }

    if (gpu) {
        double *cu_YshR, *cu_YshI;
        double *cu_YffR, *cu_YffI, *cu_YftR, *cu_YftI, *cu_YttR, *cu_YttI, *cu_YtfR, *cu_YtfI;
        double *cu_frbound, *cu_tobound;

        cudaMalloc(&cu_YshR, sizeof(double)*nw->nbus);
        cudaMalloc(&cu_YshI, sizeof(double)*nw->nbus);
        cudaMalloc(&cu_YffR, sizeof(double)*nw->active_nbranch);
        cudaMalloc(&cu_YffI, sizeof(double)*nw->active_nbranch);
        cudaMalloc(&cu_YftR, sizeof(double)*nw->active_nbranch);
        cudaMalloc(&cu_YftI, sizeof(double)*nw->active_nbranch);
        cudaMalloc(&cu_YttR, sizeof(double)*nw->active_nbranch);
        cudaMalloc(&cu_YttI, sizeof(double)*nw->active_nbranch);
        cudaMalloc(&cu_YtfR, sizeof(double)*nw->active_nbranch);
        cudaMalloc(&cu_YtfI, sizeof(double)*nw->active_nbranch);
        cudaMalloc(&cu_frbound, sizeof(double)*(2*nw->active_nbranch));
        cudaMalloc(&cu_tobound, sizeof(double)*(2*nw->active_nbranch));
        cudaMemcpy(cu_YshR, YshR, sizeof(double)*nw->nbus, cudaMemcpyHostToDevice);
        cudaMemcpy(cu_YshI, YshI, sizeof(double)*nw->nbus, cudaMemcpyHostToDevice);
        cudaMemcpy(cu_YffR, YffR, sizeof(double)*nw->active_nbranch, cudaMemcpyHostToDevice);
        cudaMemcpy(cu_YffI, YffI, sizeof(double)*nw->active_nbranch, cudaMemcpyHostToDevice);
        cudaMemcpy(cu_YftR, YftR, sizeof(double)*nw->active_nbranch, cudaMemcpyHostToDevice);
        cudaMemcpy(cu_YftI, YftI, sizeof(double)*nw->active_nbranch, cudaMemcpyHostToDevice);
        cudaMemcpy(cu_YttR, YttR, sizeof(double)*nw->active_nbranch, cudaMemcpyHostToDevice);
        cudaMemcpy(cu_YttI, YttI, sizeof(double)*nw->active_nbranch, cudaMemcpyHostToDevice);
        cudaMemcpy(cu_YtfR, YtfR, sizeof(double)*nw->active_nbranch, cudaMemcpyHostToDevice);
        cudaMemcpy(cu_YtfI, YtfI, sizeof(double)*nw->active_nbranch, cudaMemcpyHostToDevice);
        cudaMemcpy(cu_frbound, frbound, sizeof(double)*(2*nw->active_nbranch), cudaMemcpyHostToDevice);
        cudaMemcpy(cu_tobound, tobound, sizeof(double)*(2*nw->active_nbranch), cudaMemcpyHostToDevice);

        (*_YshR) = cu_YshR;
        (*_YshI) = cu_YshI;
        (*_YffR) = cu_YffR;
        (*_YffI) = cu_YffI;
        (*_YftR) = cu_YftR;
        (*_YftI) = cu_YftI;
        (*_YttR) = cu_YttR;
        (*_YttI) = cu_YttI;
        (*_YtfR) = cu_YtfR;
        (*_YtfI) = cu_YtfI;
        (*_frbound) = cu_frbound;
        (*_tobound) = cu_tobound;
    } else {
        (*_YshR) = YshR;
        (*_YshI) = YshI;
        (*_YffR) = YffR;
        (*_YffI) = YffI;
        (*_YftR) = YftR;
        (*_YftI) = YftI;
        (*_YttR) = YttR;
        (*_YttI) = YttI;
        (*_YtfR) = YtfR;
        (*_YtfI) = YtfI;
        (*_frbound) = frbound;
        (*_tobound) = tobound;
    }

    return;
}

void init_values(Network *nw, int pg_start, int qg_start, int pij_start, int qij_start,
                 int pji_start, int qji_start, int wi_i_ij_start, int wi_j_ji_start,
                 double rho_pq, double rho_va,
                 double *u_curr, double *v_curr, double *l_curr, double *rho, double *wRIij)
{
    for (int i = 0; i < nw->active_ngen; i++) {
        v_curr[pg_start+i] = 0.5*(nw->gen[i].pb / nw->baseMVA + nw->gen[i].pt / nw->baseMVA);
        v_curr[qg_start+i] = 0.5*(nw->gen[i].qb / nw->baseMVA + nw->gen[i].qt / nw->baseMVA);
    }

    for (int i = 0; i < nw->active_nbranch; i++) {
        int fr = nw->b2i[nw->branch[i].fr];
        int to = nw->b2i[nw->branch[i].to];
        double wij0 = (nw->bus[fr].nvhi*nw->bus[fr].nvhi + nw->bus[fr].nvlo*nw->bus[fr].nvlo) / 2;
        double wji0 = (nw->bus[to].nvhi*nw->bus[to].nvhi + nw->bus[to].nvlo*nw->bus[to].nvlo) / 2;
        double wR0 = sqrt(wij0 * wji0);

        u_curr[pij_start+i] = nw->YffR[i] * wij0 + nw->YftR[i] * wR0;
        u_curr[qij_start+i] = -nw->YffI[i] * wij0 - nw->YftI[i] * wR0;
        u_curr[wi_i_ij_start+i] = wij0;
        u_curr[pji_start+i] = nw->YttR[i] * wji0 + nw->YtfR[i] * wR0;
        u_curr[qji_start+i] = -nw->YttI[i] * wji0 - nw->YtfI[i] * wR0;
        u_curr[wi_j_ji_start+i] = wji0;
        //wRIij[2*i] = wR0
        //wRIij[2*i+1] = 0.0

        v_curr[wi_i_ij_start+i] = 1.0;
        v_curr[wi_j_ji_start+i] = 1.0;
    }

    memset(l_curr, 0, sizeof(double)*(2*nw->active_ngen + 6*nw->active_nbranch));
    for (int i = 0; i < 2*nw->active_ngen + 4*nw->active_nbranch; i++) {
        rho[i] = rho_pq;
    }
    for (int i = 2*nw->active_ngen + 4*nw->active_nbranch; i < 2*nw->active_ngen + 6*nw->active_nbranch; i++) {
        rho[i] = rho_va;
    }

    return;
}

void copy_data(int n, double *dest, double *src)
{
    for (int i = 0; i < n; i++) {
        dest[i] = src[i];
    }
    return;
}

void update_multiplier(int n, double *u_curr, double *v_curr,
                       double *l_curr, double *rho)
{
    for (int i = 0; i < n; i++) {
        l_curr[i] += rho[i] * (u_curr[i] - v_curr[i]);
    }
    return;
}

void primal_residual(int n, double *rp, double *u_curr, double *v_curr)
{
    for (int i = 0; i < n; i++) {
        rp[i] = u_curr[i] - v_curr[i];
    }
    return;
}

void dual_residual(int n, double *rd, double *v_prev, double *v_curr, double *rho)
{
    for (int i = 0; i < n; i++) {
        rd[i] = -rho[i] * (v_curr[i] - v_prev[i]);
    }
    return;
}

double norm(int n, double *x)
{
    double v = 0.0;
    for (int i = 0; i < n; i++) {
        v += x[i]*x[i];
    }
    v = sqrt(v);
    return v;
}

__global__
void copy_data_kernel(int n, double *dest, double *src)
{
    int tid = threadIdx.x + (blockDim.x * blockIdx.x);

    if (tid < n) {
        dest[tid] = src[tid];
    }
    return;
}

__global__
void update_multiplier_kernel(int n, double *u_curr, double *v_curr,
                              double *l_curr, double *rho)
{
    int tid = threadIdx.x + (blockDim.x * blockIdx.x);

    if (tid < n) {
        l_curr[tid] += rho[tid] * (u_curr[tid] - v_curr[tid]);
    }
    return;
}

__global__
void primal_residual_kernel(int n, double *rp, double *u_curr, double *v_curr)
{
    int tid = threadIdx.x + (blockDim.x * blockIdx.x);

    if (tid < n) {
        rp[tid] = u_curr[tid] - v_curr[tid];
    }

    return;
}

__global__
void dual_residual_kernel(int n, double *rd, double *v_prev, double *v_curr, double *rho)
{
    int tid = threadIdx.x + (blockDim.x * blockIdx.x);

    if (tid < n) {
        rd[tid] = -rho[tid] * (v_curr[tid] - v_prev[tid]);
    }

    return;
}

__global__
void norm_kernel(int n, double *x, double *v)
{
    int tid = threadIdx.x + (blockDim.x * blockIdx.x);

    double val = 0;
    if (tid < n) {
        val = x[tid]*x[tid];
    }
}

int main(int argc, char **argv)
{
    if (argc < 6) {
        usage(argv[0]);
        exit(-1);
    }

    Network *nw = nw_alloc();
    int rc = nw_build_from_matpower(nw, argv[1]);
    if (rc != OK) {
        printf("Error: parsing has failed with rc %d."
               " The model may not be for OPF.\n", rc);
        return rc;
    }

    int iterlim = atoi(argv[2]);
    double rho_pq = atof(argv[3]);
    double rho_va = atof(argv[4]);
    bool use_gpu = (atoi(argv[5]) == 1) ? true : false;

    print_mat_stat(nw);
    admittance(nw);

    int n, Kf, Kf_mean;
    double mu_max, rho_max, rho_min_pq, rho_min_w, eps_rp, eps_rp_min;
    double rt_inc, rt_dec, eta;

    n = 8;
    mu_max = 1e8;
    rho_max = 1e6;
    rho_min_pq = 5.0;
    rho_min_w = 5.0;
    eps_rp = 1e-4;
    eps_rp_min = 1e-5;
    rt_inc = 2.0;
    rt_dec = 2.0;
    eta = 0.99;
    Kf = 100;
    Kf_mean = 10;

    double *pgmin, *pgmax, *qgmin, *qgmax, *c2, *c1, *c0;
    double *cu_pgmin, *cu_pgmax, *cu_qgmin, *cu_qgmax, *cu_c2, *cu_c1, *cu_c0;

    get_generator_data(nw, &pgmin, &pgmax, &qgmin, &qgmax, &c2, &c1, &c0);
    get_generator_data(nw, &cu_pgmin, &cu_pgmax, &cu_qgmin, &cu_qgmax, &cu_c2, &cu_c1, &cu_c0, true);

    int *fr_idx, *to_idx, *gen_idx, *fr_start, *to_start, *gen_start;
    int *cu_fridx, *cu_toidx, *cu_genidx, *cu_frstart, *cu_tostart, *cu_genstart;
    double *pd, *qd;
    double *cu_pd, *cu_qd;

    get_bus_data(nw, &fr_idx, &to_idx, &gen_idx, &fr_start, &to_start, &gen_start, &pd, &qd);
    get_bus_data(nw, &cu_fridx, &cu_toidx, &cu_genidx, &cu_frstart, &cu_tostart, &cu_genstart, &cu_pd, &cu_qd, true);

    double *YshR, *YshI, *YffR, *YffI, *YftR, *YftI, *YttR, *YttI, *YtfR, *YtfI, *frbound, *tobound;
    double *cu_YshR, *cu_YshI, *cu_YffR, *cu_YffI, *cu_YftR, *cu_YftI, *cu_YttR, *cu_YttI, *cu_YtfR, *cu_YtfI, *cu_frbound, *cu_tobound;

    get_branch_data(nw, &YshR, &YshI, &YffR, &YffI, &YftR, &YftI, &YttR, &YttI, &YtfR, &YtfI, &frbound, &tobound);
    get_branch_data(nw, &cu_YshR, &cu_YshI, &cu_YffR, &cu_YffI, &cu_YftR, &cu_YftI, &cu_YttR, &cu_YttI, &cu_YtfR, &cu_YtfI, &cu_frbound, &cu_tobound, true);

    int nvars, pg_start, qg_start, pij_start, qij_start, pji_start, qji_start, wi_i_ij_start, wi_j_ji_start;
    double *u_curr, *v_curr, *l_curr, *u_prev, *v_prev, *l_prev, *rho, *param, *wRIij;
    double *rp, *rd;
    double *cu_u_curr, *cu_v_curr, *cu_l_curr, *cu_u_prev, *cu_v_prev, *cu_l_prev, *cu_rho, *cu_param, *cu_wRIij;
    double *cu_rp, *cu_rd;

    nvars = 2*nw->active_ngen + 6*nw->active_nbranch;
    pg_start = 0;
    qg_start = pg_start + nw->active_ngen;
    pij_start = qg_start + nw->active_ngen;
    qij_start = pij_start + nw->active_nbranch;
    pji_start = qij_start + nw->active_nbranch;
    qji_start = pji_start + nw->active_nbranch;
    wi_i_ij_start = qji_start + nw->active_nbranch;
    wi_j_ji_start = wi_i_ij_start + nw->active_nbranch;

    u_curr = (double *)calloc(nvars, sizeof(double));
    v_curr = (double *)calloc(nvars, sizeof(double));
    l_curr = (double *)calloc(nvars, sizeof(double));
    u_prev = (double *)calloc(nvars, sizeof(double));
    v_prev = (double *)calloc(nvars, sizeof(double));
    l_prev = (double *)calloc(nvars, sizeof(double));
    rho = (double *)calloc(nvars, sizeof(double));
    param = (double *)calloc(24*nw->active_nbranch, sizeof(double));
    wRIij = (double *)calloc(2*nw->active_nbranch, sizeof(double));
    rp = (double *)calloc(nvars, sizeof(double));
    rd = (double *)calloc(nvars, sizeof(double));

    init_values(nw, pg_start, qg_start, pij_start, qij_start, pji_start, qji_start, wi_i_ij_start, wi_j_ji_start,
                rho_pq, rho_va, u_curr, v_curr, l_curr, rho, wRIij);

    cudaMalloc(&cu_u_curr, sizeof(double)*nvars);
    cudaMalloc(&cu_v_curr, sizeof(double)*nvars);
    cudaMalloc(&cu_l_curr, sizeof(double)*nvars);
    cudaMalloc(&cu_u_prev, sizeof(double)*nvars);
    cudaMalloc(&cu_v_prev, sizeof(double)*nvars);
    cudaMalloc(&cu_l_prev, sizeof(double)*nvars);
    cudaMalloc(&cu_rho, sizeof(double)*nvars);
    cudaMalloc(&cu_param, sizeof(double)*(24*nw->active_nbranch));
    cudaMalloc(&cu_wRIij, sizeof(double)*(2*nw->active_nbranch));
    cudaMalloc(&cu_rp, sizeof(double)*nvars);
    cudaMalloc(&cu_rd, sizeof(double)*nvars);
    cudaMemcpy(cu_u_curr, u_curr, sizeof(double)*nvars, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_v_curr, v_curr, sizeof(double)*nvars, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_l_curr, l_curr, sizeof(double)*nvars, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_u_prev, u_prev, sizeof(double)*nvars, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_v_prev, v_prev, sizeof(double)*nvars, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_l_prev, l_prev, sizeof(double)*nvars, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_rho, rho, sizeof(double)*nvars, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_param, param, sizeof(double)*(24*nw->active_nbranch), cudaMemcpyHostToDevice);
    cudaMemcpy(cu_wRIij, wRIij, sizeof(double)*(2*nw->active_nbranch), cudaMemcpyHostToDevice);

    int it, max_auglag, shmem;
    int n_tb_gen, n_tb_bus, n_tb_branch;

    it = 0;
    max_auglag = 50;
    n_tb_gen = (nw->active_ngen - 1) / 32 + 1;
    n_tb_bus = (nw->nbus - 1) / 32 + 1;
    n_tb_branch = nw->active_nbranch;
    shmem = sizeof(double)*(14*n + 3*n*n) + sizeof(int)*(4*n);

    while (it < iterlim) {
        it++;

        if (!use_gpu) {
            copy_data(nvars, u_prev, u_curr);
            copy_data(nvars, v_prev, v_curr);
            copy_data(nvars, l_prev, l_curr);

            update_generator(nw->baseMVA, nw->active_ngen, pg_start, qg_start,
                             u_curr, v_curr, l_curr, rho,
                             pgmin, pgmax, qgmin, qgmax, c2, c1, c0);
            auglag(nw->active_nbranch, n, it, max_auglag, pij_start, qij_start, pji_start, qji_start,
                   wi_i_ij_start, wi_j_ji_start, mu_max,
                   u_curr, v_curr, l_curr, rho, wRIij,
                   param, YffR, YffI, YftR, YftI, YttR, YttI,
                   YtfR, YtfI, frbound, tobound);
            update_bus(nw->baseMVA, nw->nbus, pg_start, qg_start, pij_start, qij_start,
                       pji_start, qji_start, wi_i_ij_start, wi_j_ji_start,
                       fr_start, fr_idx, to_start, to_idx, gen_start, gen_idx,
                       pd, qd, u_curr, v_curr, l_curr, rho, YshR, YshI);

            update_multiplier(nvars, u_curr, v_curr, l_curr, rho);
            primal_residual(nvars, rp, u_curr, v_curr);
            dual_residual(nvars, rd, v_prev, v_curr, rho);

            printf("%10d\t%.6e\t%.6e\n", it, norm(nvars, rp), norm(nvars, rd));
        } else {
            copy_data_kernel<<<(nvars-1)/64+1, 64>>>(nvars, cu_u_prev, cu_u_curr);
            copy_data_kernel<<<(nvars-1)/64+1, 64>>>(nvars, cu_v_prev, cu_v_curr);
            copy_data_kernel<<<(nvars-1)/64+1, 64>>>(nvars, cu_l_prev, cu_l_curr);

            update_generator_kernel<<<n_tb_gen, 32>>>(nw->baseMVA, nw->active_ngen, pg_start, qg_start,
                                                      cu_u_curr, cu_v_curr, cu_l_curr, cu_rho,
                                                      cu_pgmin, cu_pgmax, cu_qgmin, cu_qgmax, cu_c2, cu_c1, cu_c0);
            auglag_kernel<<<n_tb_branch, dim3(n,n), shmem>>>(nw->active_nbranch, n, it, max_auglag, pij_start, qij_start, pji_start, qji_start,
                                                            wi_i_ij_start, wi_j_ji_start, mu_max,
                                                            cu_u_curr, cu_v_curr, cu_l_curr, cu_rho, cu_wRIij,
                                                            cu_param, cu_YffR, cu_YffI, cu_YftR, cu_YftI, cu_YttR, cu_YttI,
                                                            cu_YtfR, cu_YtfI, cu_frbound, cu_tobound);
            cudaDeviceSynchronize();

            update_bus_kernel<<<n_tb_bus, 32>>>(nw->baseMVA, nw->nbus, pg_start, qg_start, pij_start, qij_start,
                                                pji_start, qji_start, wi_i_ij_start, wi_j_ji_start,
                                                cu_frstart, cu_fridx, cu_tostart, cu_toidx, cu_genstart, cu_genidx,
                                                cu_pd, cu_qd, cu_u_curr, cu_v_curr, cu_l_curr, cu_rho, cu_YshR, cu_YshI);
            cudaDeviceSynchronize();

            update_multiplier_kernel<<<(nvars-1)/64+1, 64>>>(nvars, cu_u_curr, cu_v_curr, cu_l_curr, cu_rho);
            primal_residual_kernel<<<(nvars-1)/64+1, 64>>>(nvars, cu_rp, cu_u_curr, cu_v_curr);
            dual_residual_kernel<<<(nvars-1)/64+1, 64>>>(nvars, cu_rd, cu_v_prev, cu_v_curr, cu_rho);
            cudaDeviceSynchronize();

            printf("%10d\n", it);
        }
    }

    if (use_gpu) {
        cudaMemcpy(u_prev, cu_u_prev, sizeof(double)*nvars, cudaMemcpyDeviceToHost);
        cudaMemcpy(u_curr, cu_u_curr, sizeof(double)*nvars, cudaMemcpyDeviceToHost);
        cudaMemcpy(v_curr, cu_v_curr, sizeof(double)*nvars, cudaMemcpyDeviceToHost);
        cudaMemcpy(v_prev, cu_v_prev, sizeof(double)*nvars, cudaMemcpyDeviceToHost);
        cudaMemcpy(rho, cu_rho, sizeof(double)*nvars, cudaMemcpyDeviceToHost);
        primal_residual(nvars, rp, u_curr, v_curr);
        dual_residual(nvars, rd, v_prev, v_curr, rho);

        printf("%10d\t%.6e\t%.6e\n", it, norm(nvars, rp), norm(nvars, rd));
    }

    double objval = 0;
    for (int i = 0; i < nw->active_ngen; i++) {
        objval += c2[i]*((nw->baseMVA*u_curr[pg_start+i])*(nw->baseMVA*u_curr[pg_start+i])) +
                  c1[i]*(nw->baseMVA*u_curr[pg_start+i]) + c0[i];
    }
    printf("Objval = %e\n", objval);

    free(pgmin);
    free(pgmax);
    free(qgmin);
    free(qgmax);
    free(c2);
    free(c1);
    free(c0);

    free(fr_idx);
    free(to_idx);
    free(gen_idx);
    free(fr_start);
    free(to_start);
    free(gen_start);
    free(pd);
    free(qd);

    free(YshR);
    free(YshI);
    free(YffR);
    free(YffI);
    free(YftR);
    free(YftI);
    free(YttR);
    free(YttI);
    free(YtfR);
    free(YtfI);
    free(frbound);
    free(tobound);

    free(u_curr);
    free(v_curr);
    free(l_curr);
    free(u_prev);
    free(v_prev);
    free(l_prev);
    free(rho);
    free(param);
    free(wRIij);
    free(rp);
    free(rd);

    cudaFree(cu_pgmin);
    cudaFree(cu_pgmax);
    cudaFree(cu_qgmin);
    cudaFree(cu_qgmax);
    cudaFree(cu_c2);
    cudaFree(cu_c1);
    cudaFree(cu_c0);

    cudaFree(cu_fridx);
    cudaFree(cu_toidx);
    cudaFree(cu_genidx);
    cudaFree(cu_frstart);
    cudaFree(cu_tostart);
    cudaFree(cu_genstart);
    cudaFree(cu_pd);
    cudaFree(cu_qd);

    cudaFree(cu_YshR);
    cudaFree(cu_YshI);
    cudaFree(cu_YffR);
    cudaFree(cu_YffI);
    cudaFree(cu_YftR);
    cudaFree(cu_YftI);
    cudaFree(cu_YttR);
    cudaFree(cu_YttI);
    cudaFree(cu_YtfR);
    cudaFree(cu_YtfI);
    cudaFree(cu_frbound);
    cudaFree(cu_tobound);

    cudaFree(cu_u_curr);
    cudaFree(cu_v_curr);
    cudaFree(cu_l_curr);
    cudaFree(cu_u_prev);
    cudaFree(cu_v_prev);
    cudaFree(cu_l_prev);
    cudaFree(cu_rho);
    cudaFree(cu_param);
    cudaFree(cu_wRIij);
    cudaFree(cu_rp);
    cudaFree(cu_rd);

    return 0;
}