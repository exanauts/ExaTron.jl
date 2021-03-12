__device__
void cdriver(int n, int max_feval, int max_minor,
             int *_status, int *_minor_iter,
             double *x, double *xl, double *xu,
             double (*eval_f)(int n, double *x),
             void (*eval_g)(int n, double *x, double *g),
             void (*eval_h)(int n, double *x, double *H))
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int task, status, cg_itermax;
    int nfev, ngev, nhev, minor_iter;
    double f, delta, fatol, frtol, fmin, gtol, cgtol, gnorm0, gnorm;

    extern __shared__ double shmem[];
    double *g, *xc, *s, *wa, *gfree, *dsave;
    double *A, *B, *L;
    int *indfree, *iwa, *isave;

    g = shmem + 3*n;
    xc = g + n;
    s = xc + n;
    wa = s + n;
    gfree = wa + 6*n;
    dsave = gfree + n;
    indfree = (int *)(dsave + n);
    iwa = indfree + n;
    isave = iwa + 2*n;
    A = (double *)(isave + n);
    B = A + n*n;
    L = B + n*n;

    if (tx < n && ty == 0) {
        #pragma unroll
        for (int j = 0; j < n; j++) {
            A[n*j + tx] = 0;
            B[n*j + tx] = 0;
            L[n*j + tx] = 0;
        }
    }

    /*
    A[n*ty + tx] = 0;
    B[n*ty + tx] = 0;
    L[n*ty + tx] = 0;
    */

    cmid(n, x, xl, xu);

    task = 0;
    status = 0;
    minor_iter = 0;
    nfev = ngev = nhev = 0;

    delta = 0.0;
    fatol = 0.0;
    frtol = 1e-12;
    fmin = -1e32;
    gtol = 1e-6;
    cgtol = 0.1;
    cg_itermax = n;

    bool search = true;
    while (search) {

        // [0|1]: Evaluate function.

        if (task == 0 || task == 1) {
            f = eval_f(n, x);
            nfev += 1;
            if (nfev >= max_feval) {
                search = false;
            }
        }

        // [2] G or H: Evaluate gradient and Hessian.

        if (task == 0 || task == 2) {
            eval_g(n, x, g);
            eval_h(n, x, A);
            ngev += 1;
            nhev += 1;
            minor_iter += 1;
        }

        // Initialize the trust region bound.

        if (task == 0) {
            gnorm0 = cnrm2(n, g);
            delta = gnorm0;
        }

        // Call TRON.

        if (search) {
            ctron(n, x, xl, xu, f, g, A, frtol, fatol, fmin, cgtol, cg_itermax,
                  &delta, &task, B, L, xc, s, indfree, gfree, isave, dsave, wa, iwa);
        }

        // [3] NEWX: a new point was computed.

        if (task == 3) {
            gnorm = cgpnorm(n, x, xl, xu, g);
            if (gnorm <= gtol) {
                task = 4;
            }

            if (minor_iter >= max_minor) {
                status = 1;
                search = false;
            }
        }

        // [4] CONV: convergence was achieved.

        if (task == 4) {
            search = false;
        }
    }

    (*_status) = status;
    (*_minor_iter) = minor_iter;

    return;
}

__device__
void cdriver_auglag(int n, int max_feval, int max_minor,
                    int *_status, int *_minor_iter,
                    double *x, double *xl, double *xu,
                    double *param,
                    double YffR, double YffI,
                    double YftR, double YftI,
                    double YttR, double YttI,
                    double YtfR, double YtfI,
                    double (*eval_f)(int n, double *x, double *param,
                                    double YffR, double YffI,
                                    double YftR, double YftI,
                                    double YttR, double YttI,
                                    double YtfR, double YtfI),
                    void (*eval_g)(int n, double *x, double *g, double *param,
                                    double YffR, double YffI,
                                    double YftR, double YftI,
                                    double YttR, double YttI,
                                    double YtfR, double YtfI),
                    void (*eval_h)(int n, double *x, double *H, double *param,
                                    double YffR, double YffI,
                                    double YftR, double YftI,
                                    double YttR, double YttI,
                                    double YtfR, double YtfI))
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int task, status, cg_itermax;
    int nfev, ngev, nhev, minor_iter;
    double f, delta, fatol, frtol, fmin, gtol, cgtol, gnorm0, gnorm;

    extern __shared__ double shmem[];
    double *g, *xc, *s, *wa, *gfree, *dsave;
    double *A, *B, *L;
    int *indfree, *iwa, *isave;

    g = shmem + 3*n;
    xc = g + n;
    s = xc + n;
    wa = s + n;
    gfree = wa + 6*n;
    dsave = gfree + n;
    indfree = (int *)(dsave + n);
    iwa = indfree + n;
    isave = iwa + 2*n;
    A = (double *)(isave + n);
    B = A + n*n;
    L = B + n*n;

    if (tx < n && ty == 0) {
        #pragma unroll
        for (int j = 0; j < n; j++) {
            A[n*j + tx] = 0;
            B[n*j + tx] = 0;
            L[n*j + tx] = 0;
        }
    }
    /*
    A[n*ty + tx] = 0;
    B[n*ty + tx] = 0;
    L[n*ty + tx] = 0;
    */
    cmid(n, x, xl, xu);  // __syncthreads() will be called in cmid().

    task = 0;
    status = 0;

    delta = 0.0;
    fatol = 0.0;
    frtol = 1e-12;
    fmin = -1e32;
    gtol = 1e-6;
    cgtol = 0.1;
    cg_itermax = n;

    f = 0.0;
    nfev = 0;
    ngev = 0;
    nhev = 0;
    minor_iter = 0;

    bool search = true;
    while (search) {

        // [0|1]: Evaluate function.

        if (task == 0 || task == 1) {
            f = eval_f(n, x, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI);
            nfev += 1;
            if (nfev >= max_feval) {
                search = false;
            }
        }

        // [2] G or H: Evaluate gradient and Hessian.

        if (task == 0 || task == 2) {
            eval_g(n, x, g, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI);
            eval_h(n, x, A, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI);
            ngev += 1;
            nhev += 1;
            minor_iter += 1;
        }

        // Initialize the trust region bound.

        if (task == 0) {
            gnorm0 = cnrm2(n, g);
            delta = gnorm0;
        }

        // __syncthreads(); // false positive

        // Call TRON.

        if (search) {
            ctron(n, x, xl, xu, f, g, A, frtol, fatol, fmin, cgtol, cg_itermax,
                  &delta, &task, B, L, xc, s, indfree, gfree, isave, dsave, wa, iwa);
        }

        // [3] NEWX: a new point was computed.

        if (task == 3) {
            gnorm = cgpnorm(n, x, xl, xu, g);
            if (gnorm <= gtol) {
                task = 4;
            }

            if (minor_iter >= max_minor) {
                status = 1;
                search = false;
            }
        }

        // [4] CONV: convergence was achieved.

        if (task == 4 || task == 10) {
            search = false;
        }
    }

    (*_status) = status;
    (*_minor_iter) = minor_iter;

    __syncthreads();

    return;
}