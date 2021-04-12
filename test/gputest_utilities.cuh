double nrm2_vec(int n, double *x)
{
    double v = 0;
    for (int j = 0; j < n; j++) {
        v += x[j]*x[j];
    }
    return sqrt(v);
}

void nrm2_mat(int n, double *wa, double *A)
{
    for (int j = 0; j < n; j++) {
        wa[j] = A[n*j + j]*A[n*j + j];
    }

    for (int j = 0; j < n; j++) {
        for (int i = j+1; i < n; i++) {
            double v = A[n*j + i]*A[n*j + i];
            wa[j] += v;
            wa[i] += v;
        }
    }

    for (int j = 0; j < n; j++) {
        wa[j] = sqrt(wa[j]);
    }

    return;
}

double gpnorm(int n, double *x, double *xl, double *xu, double *g)
{
    double max_v = 0;
    for (int j = 0; j < n; j++) {
        double v = 0;
        if (xl[j] != xu[j]) {
            if (x[j] == xl[j]) {
                v = min(g[j], 0.0);
            } else if (x[j] == xu[j]) {
                v = max(g[j], 0.0);
            } else {
                v = g[j];
            }

            v = abs(v);
        }
        max_v = max(max_v, v);
    }

    return max_v;
}

// Left-looking Cholesky factorization.
int icf_left(int n, double *L)
{
    int info = 0;

    for (int j = 0; j < n; j++) {
        if (L[n*j + j] <= 0) {
            info = -j;
            break;
        }

        // Update the jth column and diagonals of the trailing matrix.
        L[n*j + j] = sqrt(L[n*j + j]);
        for (int k = 0; k < j; k++) {
            for (int i = j+1; i < n; i++) {
                L[n*j + i] -= L[n*k + i]*L[n*k + j];
            }
        }

        for (int i = j+1; i < n; i++) {
            L[n*j + i] /= L[n*j + j];
            L[n*i + i] -= L[n*j + i]*L[n*j + i];
        }
    }

    return info;
}

// Right-looking Cholesky factorization.
int icf_right(int n, double *L)
{
    int info = 0;

    for (int j = 0; j < n; j++) {
        // Update the diagonal.
        if (L[n*j + j] > 0) {
            L[n*j + j] = sqrt(L[n*j + j]);
        } else {
            info = -1;
            break;
        }

        // Update the jth column.
        for (int i = j+1; i < n; i++) {
            L[n*j + i] /= L[n*j + j];
        }

        // Update the trailing matrix.
        for (int c = j+1; c < n; c++) {
            for (int r = c; r < n; r++) {
                L[n*c + r] -= (L[n*j + c]*L[n*j + r]);
            }
        }
    }

    return info;
}

void icfs(int n, double alpha, double *A, double *L, double *wa1, double *wa2)
{
    int nbmax = 3, nbfactor = 512;

    // Compute the l2 norms of the columns of A.
    nrm2_mat(n, wa1, A);

    // Compute the scaling matrix D.
    for (int j = 0; j < n; j++) {
        wa2[j] = (wa1[j] > 0.0) ? 1.0/sqrt(wa1[j]) : 1.0;
    }

    // Determine a lower bound for the step.
    double alphas = (alpha <= 0.0) ? 1.0e-3 : alpha;

    // Compute the initial shift.
    alpha = 0.0;
    for (int j = 0; j < n; j++) {
        alpha = (A[n*j + j] == 0.0) ? alphas : max(alpha, -A[n*j + j]*(wa2[j]*wa2[j]));
    }

    if (alpha > 0) {
        alpha = max(alpha, alphas);
    }

    int nb = 1, info = 0;
    while (1) {
        for (int j = 0; j < n; j++) {
            for (int k = j; k < n; k++) {
                L[n*j + k] = A[n*j + k]*(wa2[j]*wa2[k]);
            }
        }
        if (alpha != 0.0) {
            for (int j = 0; j < n; j++) {
                L[n*j + j] += alpha;
            }
        }

        // Attempt a Cholesky factorization.
        info = icf_right(n, L);

        // If the factorization exists, then test for termination.
        // Otherwise, increment the shift.
        if (info >= 0) {
            // If the shift is at the lower bound, reduce the shift.
            // Otherwise, undo the scaling of L and exit.
            if ((alpha == alphas) && (nb < nbmax)) {
                alphas /= nbfactor;
                alpha = alphas;
                nb++;
            } else {
                for (int j = 0; j < n; j++) {
                    for (int k = j; k < n; k++) {
                        L[n*j + k] /= wa2[k];
                        L[n*k + j] = L[n*j + k];
                    }
                }
                return;
            }
        } else {
            alpha = max(2.0*alpha, alphas);
        }
    }

    return;
}

void llt(int n, double *A, double *L)
{
    memset(A, 0, (n*n)*sizeof(double));

    // A = L*L^T
    for (int k = 0; k < n; k++) {
        for (int j = 0; j <= k; j++) {
            double s = L[n*j + k];
            for (int i = j; i < n; i++) {
                A[n*k + i] += s*L[n*j + i];
            }
        }
    }

    // Make sure A is symmetric by setting A[j,i] = A[i,j] with i > j.
    for (int j = 0; j < n; j++) {
        for (int i = j+1; i < n; i++) {
            A[n*i + j] = A[n*j + i];
        }
    }

    return;
}

void copy(int n, double *src, double *dest)
{
    for (int i = 0; i < n; i++) {
        dest[i] = src[i];
    }
    return;
}

void scal(int n, double s, double *w)
{
    for (int i = 0; i < n; i++) {
        w[i] *= s;
    }
    return;
}

double dot(int n, double *a, double *b)
{
    double v = 0;
    for (int i = 0; i < n; i++) {
        v += a[i]*b[i];
    }
    return v;
}

void breakpt(int n, double *x, double *xl, double *xu, double *w,
    int *_nbrpt, double *_brptmin, double *_brptmax)
{
    int nbrpt = 0;
    double brpt, brptmin = 0.0, brptmax = 0.0;

    for (int i = 0; i < n; i++) {
        if (x[i] < xu[i] && w[i] > 0.0) {
            nbrpt += 1;
            brpt = (xu[i] - x[i]) / w[i];
            if (nbrpt == 1) {
                brptmin = brpt;
                brptmax = brpt;
            } else {
                brptmin = min(brpt, brptmin);
                brptmax = max(brpt, brptmax);
            }
        } else if (x[i] > xl[i] && w[i] < 0.0) {
            nbrpt += 1;
            brpt = (xl[i] - x[i]) / w[i];
            if (nbrpt == 1) {
                brptmin = brpt;
                brptmax = brpt;
            } else {
                brptmin = min(brpt, brptmin);
                brptmax = max(brpt, brptmax);
            }
        }
    }

    // Hande the exceptional case.
    if (nbrpt == 0) {
        brptmin = 0.0;
        brptmax = 0.0;
    }

    (*_nbrpt) = nbrpt;
    (*_brptmin) = brptmin;
    (*_brptmax) = brptmax;

    return;
}

void axpy(int n, double alpha, double *x, double *y)
{
    for (int j = 0; j < n; j++) {
        y[j] += alpha*x[j];
    }
    return;
}

void gemv(int n, double *z, double alpha, double *A, double *x, double beta, double *y)
{
    memset(z, 0, sizeof(double)*n);
    for (int j = 0; j < n; j++) {
        for (int k = 0; k < n; k++) {
            z[k] += alpha*(A[n*j + k]*x[j]);
        }
    }

    if (beta != 0 && y != NULL) {
        for (int k = 0; k < n; k++) {
            z[k] += beta*y[k];
        }
    }

    return;
}

double diff_matrix(int n, double *A, double *B)
{
    double err = 0;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            err = max(err, fabs(A[n*j+i] - B[n*j+i]));
        }
    }
    return err;
}

double diff_vector(int n, double *a, double *b)
{
    double err = 0;
    for (int j = 0; j < n; j++) {
        err = max(err, abs(a[j] - b[j]));
    }
    return err;
}

void mid(int n, double *x, double *xl, double *xu)
{
    for (int j = 0; j < n; j++) {
        x[j] = max(xl[j], min(xu[j], x[j]));
    }
    return;
}

void gpstep(int n, double *x, double *xl, double *xu, double alpha, double *w, double *s)
{
    for (int j = 0; j < n; j++) {
        if (x[j] + alpha*w[j] < xl[j]) {
            s[j] = xl[j] - x[j];
        } else if (x[j] + alpha*w[j] > xu[j]) {
            s[j] = xu[j] - x[j];
        } else {
            s[j] = alpha*w[j];
        }
    }
    return;
}

double cauchy(int n, double *x, double *xl, double *xu,
             double *A, double *g, double delta, double alpha,
             double *s, double *wa)
{
    double gts, q, alphas;

    // Constant that defines sufficient decrease.
    double mu0 = 0.01;

    // Interpolation and extrapolation factors.
    double interpf = 0.1;
    double extrapf = 10.0;

    // Find the minimal and maximal breakpoint on x-alpha*g.
    copy(n, g, wa);
    scal(n, -1.0, wa);

    int nbrpt;
    double brptmin, brptmax;
    breakpt(n, x, xl, xu, wa, &nbrpt, &brptmin, &brptmax);

    // Evaluate the initial alpha and decide if the algorithm
    // must interpolate or extrapolate.

    bool interp;
    gpstep(n, x, xl, xu, -alpha, g, s);
    if (nrm2_vec(n, s) > delta) {
        interp = true;
    } else {
        gemv(n, wa, 1.0, A, s, 0.0, NULL);
        gts = dot(n, g, s);
        q = 0.5*dot(n, s, wa) + gts;
        interp = (q >= mu0*gts);
    }

    // Either interpolate or extrapolate to find a successful step.

    bool search = true;
    if (interp) {
        while (search) {
            alpha *= interpf;
            gpstep(n, x, xl, xu, -alpha, g, s);
            if (nrm2_vec(n, s) <= delta) {
                gemv(n, wa, 1.0, A, s, 0.0, NULL);
                gts = dot(n, g, s);
                q = 0.5*dot(n, s, wa) + gts;
                search = (q > mu0*gts);
            }
        }
    } else {
        // Increase alpha until a successful step is found.

        alphas = alpha;
        while (search && alpha <= brptmax) {
            alpha *= extrapf;
            gpstep(n, x, xl, xu, -alpha, g, s);
            if (nrm2_vec(n, s) <= delta) {
                gemv(n, wa, 1.0, A, s, 0.0, NULL);
                gts = dot(n, g, s);
                q = 0.5*dot(n, s, wa) + gts;
                if (q < mu0*gts) {
                    search = true;
                    alphas = alpha;
                }
            } else {
                search = false;
            }
        }

        // Recover the last successful step.
        alpha = alphas;
        gpstep(n, x, xl, xu, -alpha, g, s);
    }

    return alpha;
}

void prsrch(int n, double *x, double *xl, double *xu, double *A,
            double *g, double *w, double *wa1, double *wa2)
{
    double gts, q;

    // Constant that defines sufficient decrease.
    double mu0 = 0.01;

    // Interpolation factor.

    double interpf = 0.5;

    // Set the initial alpha = 1 because the quadratic function is
    // decreasing in the ray x + alpha*w for 0 <= alpha <= 1.

    double alpha = 1.0;
    int nsteps = 0;

    // Find the smallest break-point on the ray x + alpha*w.
    int nbrpt;
    double brptmin, brptmax;
    breakpt(n, x, xl, xu, w, &nbrpt, &brptmin, &brptmax);

    bool search = true;
    while (search && alpha > brptmin) {

        // Calculate P[x + alpha*w] - x and check the sufficient
        // decrease condition.

        nsteps += 1;
        gpstep(n, x, xl, xu, alpha, w, wa1);
        gemv(n, wa2, 1.0, A, wa1, 0.0, NULL);
        gts = dot(n, g, wa1);
        q = 0.5*dot(n, wa1, wa2) + gts;
        if (q <= mu0*gts) {
            search = false;
        }
        else {

            // This is a crude interpolation procedure that
            // will be replaced in future versions of the code.

            alpha = interpf*alpha;
        }
    }

    // Force at least one more constraint to be added to the active
    // set if alpha < brptmin and the full step is not successful.
    // There is sufficient decrease because the quadratic function
    // is decreasing in the ray x + alpha*w for 0 <= alpha <= 1.

    if (alpha < 1.0 && alpha < brptmin) {
        alpha = brptmin;
    }

    // Compute the final iterate and step.

    gpstep(n, x, xl, xu, alpha, w, wa1);
    axpy(n, alpha, w, x);
    mid(n, x, xl, xu);
    copy(n, wa1, w);
}

void nsol(int n, double *L, double *r)
{
    // Solve L*x = r and store the result in r.
    for (int j = 0; j < n; j++) {
        r[j] /= L[n*j + j];
        for (int k = j+1; k < n; k++) {
            r[k] -= L[n*j + k]*r[j];
        }
    }

    return;
}

void tsol(int n, double *L, double *r)
{
    // Solve L'* = r and store the result in r.
    for (int j = n-1; j >= 0; j--) {
        r[j] /= L[n*j + j];
        for (int k = 0; k < j; k++) {
            r[k] -= L[n*j + k]*r[j];
        }
    }

    return;
}

double trqsol(int n, double *x, double *p, double delta)
{
    double sigma = 0.0;

    double ptx = dot(n, p, x);
    double ptp = dot(n, p, p);
    double xtx = dot(n, x, x);
    double dsq = delta*delta;

    // Guard against abnormal cases.
    double rad = ptx*ptx + ptp*(dsq - xtx);
    rad = sqrt(max(rad, 0.0));

    if (ptx > 0.0) {
        sigma = (dsq - xtx) / (ptx + rad);
    } else if (rad > 0.0) {
        sigma = (rad - ptx) / ptp;
    } else {
        sigma = 0.0;
    }

    return sigma;
}

void trpcg(int n, double *A, double *g, double delta, double *L,
           double tol, double stol, int itermax, double *w,
           double *p, double *q, double *r, double *t, double *z,
           int *_info, int *_iters)
{
    // Initialize the iterate w and the residual r.
    for (int i = 0; i < n; i++) {
        w[i] = 0.0;
    }

    // Initialize the residual t of grad q to -g.
    // Initialize the residual r of grad Q by solving L*r = -g.
    // Note that t = L*r.

    copy(n, g, t);
    scal(n, -1.0, t);
    copy(n, t, r);
    nsol(n, L, r);

    // Initialize the direction p.
    copy(n, r, p);

    //Initialize rho and the norms of r and t.
    double rho = dot(n, r, r);
    double rnorm0 = sqrt(rho);

    // Exit if g = 0.
    int iters = 0;
    if (rnorm0 == 0.0) {
        (*_info) = 1;
        (*_iters) = iters;
        return;
    }

    double ptq, alpha, sigma, rtr, rnorm, tnorm, beta;
    for (iters = 0; iters < itermax; iters++) {

        // Note:
        // Q(w) = 0.5*w'Bw + h'w, where B=L^{-1}AL^{-T}, h=L^{-1}g.
        // Then p'Bp = p'L^{-1}AL^{-T}p = p'L^{-1}Az = p'q.
        // alpha = r'r / p'Bp.

        copy(n, p, z);
        tsol(n, L, z);

        // Compute q by solving L*q = A*z and save L*q for
        // use in updating the residual t.
        gemv(n, q, 1.0, A, z, 0.0, NULL);
        copy(n, q, z);
        nsol(n, L, q);

        // Compute alpha and determine sigma such that the trust region
        // constraint || w + sigma*p || = delta is satisfied.
        ptq = dot(n, p, q);
        if (ptq > 0.0) {
            alpha = rho / ptq;
        } else {
            alpha = 0.0;
        }

        sigma = trqsol(n, w, p, delta);

        // Exit if there is negative curvature or if the
        // iterates exit the trust region.

        if (ptq <= 0.0 || alpha >= sigma) {
            axpy(n, sigma, p, w);
            if (ptq <= 0.0) {
                (*_info) = 3;
            } else {
                (*_info) = 4;
            }

            (*_iters) = iters;
            return;
        }

        // Update w and the residuals r and t.
        // Note that t = L*r.

        axpy(n, alpha, p, w);
        axpy(n, -alpha, q, r);
        axpy(n, -alpha, z, t);

        // Exit if the residual convergence test is satisfied.

        rtr = dot(n, r, r);
        rnorm = sqrt(rtr);
        tnorm = sqrt(dot(n, t, t));

        if (tnorm <= tol) {
            (*_info) = 1;
            (*_iters) = iters;
            return;
        }

        if (rnorm <= stol) {
            (*_info) = 2;
            (*_iters) = iters;
            return;
        }

        // Compute p = r + beta*p and update rho.
        beta = rtr / rho;
        scal(n, beta, p);
        axpy(n, 1.0, r, p);
        rho = rtr;
    }

    iters = itermax;
    (*_info) = 5;
    (*_iters) = iters;
    return;
}

void reorder(int n, int nfree, double *B, double *A,
              int *indfree, int *iwa)
{
    int jfree;
    for (int j = 0; j < nfree; j++) {
        jfree = indfree[j];
        B[nfree*j + j] = A[n*jfree + jfree];
        for (int i = jfree+1; i < n; i++) {
            if (iwa[i] > 0) {
                B[nfree*j + iwa[i]] = A[n*jfree + i];
                B[nfree*iwa[i] + j] = B[nfree*j + iwa[i]];
            }
        }
    }

    return;
}

void spcg(int n, double delta, double rtol, int itermax,
           double *x, double *xl, double *xu, double *A, double *g,
           double *s, double *B, double *L, int *indfree, double *gfree,
           double *w, int *iwa, double *wa, int *_info, int *_iters)
{
    // Compute A*(x1 - x0) and store in w.
    gemv(n, w, 1.0, A, s, 0.0, NULL);

    // Compute the Cauchy point.
    axpy(n, 1.0, s, x);
    mid(n, x, xl, xu);

    // Start the main iteration loop.
    // There are at most n ierations because at each iteration
    // at least one variable becomes active.

    double alpha, gfnorm, gfnormf, tol, stol;
    int info = 3, iters = 0, info_tr, iters_tr;
    int nfree;
    for (int nfaces = 0; nfaces < n; nfaces++) {

        // Determine the free variables at the current minimizer.
        // The indices of the free variables are stored in the first
        // n free positions of the array indfree.
        // The array iwa is used to detect free variables by setting
        // iwa[i] = nfree if the ith variable is free, otherwise iwa[i] = 0.

        nfree = 0;
        for (int j = 0; j < n; j++) {
            if (xl[j] < x[j] && x[j] < xu[j]) {
                indfree[nfree] = j;
                iwa[j] = nfree;
                nfree++;
            } else {
                iwa[j] = 0;
            }
        }

        if (nfree == 0) {
            (*_info) = 1;
            (*_iters) = iters;
            return;
        }

        // Obtain the submatrix of A for the free variables.
        // Recall that iwa allows the detection of free variables.

        reorder(n, nfree, B, A, indfree, iwa);

        // Compute the Cholesky factorization with diagonal shifting.

        alpha = 0.0;
        icfs(nfree, alpha, B, L, wa, &wa[n]);

        // Compute the gradient grad q(x[k]) = g + A*(x[k] - x[0])
        // of q at x[k] for the free variables.
        // Recall that w contains A*(x[k] - x[0]).
        // Compute the norm of the reduced gradient Z'*g.

        for (int j = 0; j < nfree; j++) {
            gfree[j] = w[indfree[j]] + g[indfree[j]];
            wa[j] = g[indfree[j]];
        }
        gfnorm = nrm2_vec(nfree, wa);

        // Save the trust region subproblem in the free variables
        // to generate a direction p[k]. Store p[k] in the array w.

        tol = rtol*gfnorm;
        stol = 0.0;

        trpcg(nfree, B, gfree, delta, L, tol, stol, itermax,
               w, wa, &wa[n], &wa[2*n], &wa[3*n], &wa[4*n],
               &info_tr, &iters_tr);
        iters += iters_tr;
        tsol(nfree, L, w);

        // Use a projected search to obtain the next iterate.
        // The projected search algorithm stores s[k] in w.

        for (int j = 0; j < nfree; j++) {
            wa[j] = x[indfree[j]];
            wa[n + j] = xl[indfree[j]];
            wa[2*n + j] = xu[indfree[j]];
        }

        prsrch(nfree, wa, &wa[n], &wa[2*n], B, gfree, w, &wa[3*n], &wa[4*n]);

        // Update the minimizer and the step.
        // Note that s now contains x[k+1] - x[0].

        for (int j = 0; j < nfree; j++) {
            x[indfree[j]] = wa[j];
            s[indfree[j]] += w[j];
        }

        // Compute A*(x[k+1] - x[0]) and store in w.
        gemv(n, w, 1.0, A, s, 0.0, NULL);

        // Compute the gradient grad q(x[k+1]) = g + A*(x[k+1] - x[0])
        // of q at x[k+1] for the free variables.

        for (int j = 0; j < nfree; j++) {
            gfree[j] = w[indfree[j]] + g[indfree[j]];
        }

        gfnormf = nrm2_vec(nfree, gfree);

        // Convergence and termination test.
        // We terminate if the preconditioned conjugate gradient
        // method encounters a direction of negative curvature, or
        // if the step is at the trust region bound.

        if (gfnormf <= rtol*gfnorm) {
            (*_info) = 1;
            (*_iters) = iters;
            return;
        } else if (info_tr == 3 || info_tr == 4) {
            (*_info) = 2;
            (*_iters) = iters;
            return;
        } else if (iters > itermax) {
            (*_info) = 3;
            (*_iters) = iters;
            return;
        }
    }

    (*_info) = info;
    (*_iters) = iters;
    return;
}

void tron(int n, double *x, double *xl, double *xu, double f, double *g,
          double *A, double frtol, double fatol, double fmin, double cgtol,
          int itermax, double *_delta, int *_task, double *B, double *L,
          double *xc, double *s, int *indfree, double *gfree, int *isave,
          double *dsave, double *wa, int *iwa)
{
    int task, work, iter, iterscg, info, iters;
    double eta0, eta1, eta2, sigma1, sigma2, sigma3;
    double delta, alpha, alphac, fc, g0, prered, actred, snorm;
    bool search;

    // Parameters for updating the iterates.

    eta0 = 1.0e-4;
    eta1 = 0.25;
    eta2 = 0.75;

    // Parameters for updating the trust region size delta.

    sigma1 = 0.25;
    sigma2 = 0.5;
    sigma3 = 4.0;

    work = 0;

    // Initialization section.

    delta = (*_delta);
    task = (*_task);

    if (task == 0)  { // "START"

        // Initialize local variables.

        iter = 1;
        iterscg = 0;
        alphac = 1.0;
        work = 1;  // "COMPUTE"
    } else {

        // Restore local variables.

        work = isave[0];
        iter = isave[1];
        iterscg = isave[2];
        fc = dsave[0];
        alphac = dsave[1];
        prered = dsave[2];
    }

    // Search for a lower function value.

    search = true;
    while (search) {

        // Compute a step and evaluate the function at the trial point.

        if (work == 1) { // "COMPUTE"

            // Save the best function value, iterate, and gradient.

            fc = f;
            copy(n, x, xc);

            // Compute the Cauchy step and store in s.

            alphac = cauchy(n, x, xl, xu, A, g, delta, alphac, s, wa);

            // Compute the projected Newton step.

            spcg(n, delta, cgtol, itermax, x, xl, xu, A, g, s,
                 B, L, indfree, gfree, wa, iwa, &wa[n], &info, &iters);

            // Compute the predicted reduction.

            gemv(n, wa, 1.0, A, s, 0.0, NULL);
            prered = -(dot(n, s, g) + 0.5*dot(n, s, wa));
            iterscg = iterscg + iters;

            // Set task to compute the function.

            task = 1; // 'F'
        }

        // Evaluate the step and determine if the step is successful.

        if (work == 2) { // "EVALUATE"

            // Compute the actual reduction.

            actred = fc - f;

            // On the first iteration, adjust the initial step bound.

            snorm = nrm2_vec(n, s);
            if (iter == 1)
                delta = min(delta, snorm);

            // Update the trust region bound.

            g0 = dot(n, g, s);
            if (f-fc-g0 <= 0) {
                alpha = sigma3;
            } else {
                alpha = max(sigma1, -0.5*(g0/(f-fc-g0)));
            }

            // Update the trust region bound according to the ratio
            // of actual to predicted reduction.

            if (actred < eta0*prered) {
                delta = min(max(alpha, sigma1)*snorm, sigma2*delta);
            } else if (actred < eta1*prered) {
                delta = max(sigma1*delta, min(alpha*snorm, sigma2*delta));
            } else if (actred < eta2*prered) {
                delta = max(sigma1*delta, min(alpha*snorm, sigma3*delta));
            } else {
                delta = max(delta, min(alpha*snorm, sigma3*delta));
            }

            // Update the iterate.

            if (actred > eta0*prered) {

                // Successful iterate.

                task = 2; // 'G' or 'H'
                iter += 1;
            } else {

                // Unsuccessful iterate.

                task = 1; // 'F'
                copy(n, xc, x);
                f = fc;
            }

            // Test for convergence.

            if (f < fmin) {
                task = 10; // "WARNING: F .LT. FMIN"
            }
            if (abs(actred) <= fatol && prered <= fatol) {
                task = 4;  // "CONVERGENCE: FATOL TEST SATISFIED"
            }
            if (abs(actred) <= frtol*abs(f) && prered <= frtol*abs(f)) {
                task = 4;  // "CONVERGENCE: FRTOL TEST SATISFIED"
            }
        }

        // Test for continuation of search

        if (task == 1 && work == 2) { // Char(task[1]) == 'F' && work == "EVALUATE"
            search = true;
            work = 1; // "COMPUTE"
        } else {
            search = false;
        }
    }

    if (work == 3) { // "NEWX"
        task = 3;  // "NEWX"
    }

    // Decide on what work to perform on the next iteration.

    if (task == 1 && work == 1) { // Char(task[1]) == 'F' && work == "COMPUTE"
        work = 2; // "EVALUATE"
    } else if (task == 1 && work == 2) { // Char(task[1]) == 'F' && work == "EVALUATE"
        work = 1; // "COMPUTE"
    } else if (task == 2) { // unsafe_string(pointer(task),2) == "GH"
        work = 3; // "NEWX"
    } else if (task == 3) { // unsafe_string(pointer(task),4) == "NEWX"
        work = 1; // "COMPUTE"
    }

    // Save local variables.

    isave[0] = work;
    isave[1] = iter;
    isave[2] = iterscg;

    dsave[0] = fc;
    dsave[1] = alphac;
    dsave[2] = prered;

    (*_delta) = delta;
    (*_task) = task;

    return;
}

void driver(int n, int max_feval, int max_minor,
            int *_status, int *_minor_iter,
            double *x, double *xl, double *xu,
            double (*eval_f)(int n, double *x, int bx),
            void (*eval_g)(int n, double *x, double *g, int bx),
            void (*eval_h)(int n, double *x, double *H, int bx),
            int bx)
{
    int task, status, cg_itermax;
    int nfev, ngev, nhev, minor_iter;
    double f, delta, fatol, frtol, fmin, gtol, cgtol, gnorm0, gnorm;

    double *g, *xc, *s, *wa, *gfree, *dsave;
    double *A, *B, *L;
    int *indfree, *iwa, *isave;

    g = (double *)calloc(n, sizeof(double));
    xc = (double *)calloc(n, sizeof(double));
    s = (double *)calloc(n, sizeof(double));
    wa = (double *)calloc(6*n, sizeof(double));
    gfree = (double *)calloc(n, sizeof(double));
    dsave = (double *)calloc(n, sizeof(double));
    indfree = (int *)calloc(n, sizeof(int));
    iwa = (int *)calloc(2*n, sizeof(int));
    isave = (int *)calloc(n, sizeof(int));
    A = (double *)calloc(n*n, sizeof(double));
    B = (double *)calloc(n*n, sizeof(double));
    L = (double *)calloc(n*n, sizeof(double));

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

    for (int j = 0; j < n; j++) {
        x[j] = max(xl[j], min(xu[j], x[j]));
    }

    bool search = true;
    while (search) {

        // [0|1]: Evaluate function.

        if (task == 0 || task == 1) {
            f = eval_f(n, x, bx);
            nfev += 1;
            if (nfev >= max_feval) {
                search = false;
            }
        }

        // [2] G or H: Evaluate gradient and Hessian.

        if (task == 0 || task == 2) {
            eval_g(n, x, g, bx);
            eval_h(n, x, A, bx);
            ngev += 1;
            nhev += 1;
            minor_iter += 1;
        }

        // Initialize the trust region bound.

        if (task == 0) {
            gnorm0 = nrm2_vec(n, g);
            delta = gnorm0;
        }

        // Call TRON.

        if (search) {
            tron(n, x, xl, xu, f, g, A, frtol, fatol, fmin, cgtol, cg_itermax,
                 &delta, &task, B, L, xc, s, indfree, gfree, isave, dsave, wa, iwa);
        }

        // [3] NEWX: a new point was computed.

        if (task == 3) {
            gnorm = gpnorm(n, x, xl, xu, g);
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

    free(g);
    free(xc);
    free(s);
    free(wa);
    free(gfree);
    free(dsave);
    free(indfree);
    free(iwa);
    free(isave);
    free(A);
    free(B);
    free(L);

    (*_status) = status;
    (*_minor_iter) = minor_iter;

    return;
}

void driver_auglag(int I, int n, int max_feval, int max_minor,
                   int *_status, int *_minor_iter, double scale,
                   double *x, double *xl, double *xu,
                   double *param,
                   double YffR, double YffI, double YftR, double YftI,
                   double YttR, double YttI, double YtfR, double YtfI,
                   double (*eval_f)(int I, int n, double scale, double *x, double *param,
                                    double YffR, double YffI, double YftR, double YftI,
                                    double YttR, double YttI, double YtfR, double YtfI),
                   void (*eval_g)(int I, int n, double scale, double *x, double *g, double *param,
                                  double YffR, double YffI, double YftR, double YftI,
                                  double YttR, double YttI, double YtfR, double YtfI),
                   void (*eval_h)(int I, int n, double scale, double *x, double *H, double *param,
                                  double YffR, double YffI, double YftR, double YftI,
                                  double YttR, double YttI, double YtfR, double YtfI))
{
    int task, status, cg_itermax;
    int nfev, ngev, nhev, minor_iter;
    double f, delta, fatol, frtol, fmin, gtol, cgtol, gnorm0, gnorm;

    double *g, *xc, *s, *wa, *gfree, *dsave;
    double *A, *B, *L;
    int *indfree, *iwa, *isave;

    g = (double *)calloc(n, sizeof(double));
    xc = (double *)calloc(n, sizeof(double));
    s = (double *)calloc(n, sizeof(double));
    wa = (double *)calloc(6*n, sizeof(double));
    gfree = (double *)calloc(n, sizeof(double));
    dsave = (double *)calloc(n, sizeof(double));
    indfree = (int *)calloc(n, sizeof(int));
    iwa = (int *)calloc(2*n, sizeof(int));
    isave = (int *)calloc(n, sizeof(int));
    A = (double *)calloc(n*n, sizeof(double));
    B = (double *)calloc(n*n, sizeof(double));
    L = (double *)calloc(n*n, sizeof(double));

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

    for (int j = 0; j < n; j++) {
        x[j] = max(xl[j], min(xu[j], x[j]));
    }

    bool search = true;
    while (search) {

        // [0|1]: Evaluate function.

        if (task == 0 || task == 1) {
            f = eval_f(I, n, scale, x, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI);
            nfev += 1;
            if (nfev >= max_feval) {
                search = false;
            }
        }

        // [2] G or H: Evaluate gradient and Hessian.

        if (task == 0 || task == 2) {
            eval_g(I, n, scale, x, g, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI);
            eval_h(I, n, scale, x, A, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI);
            ngev += 1;
            nhev += 1;
            minor_iter += 1;
        }

        // Initialize the trust region bound.

        if (task == 0) {
            gnorm0 = nrm2_vec(n, g);
            delta = gnorm0;
        }

        // Call TRON.

        if (search) {
            tron(n, x, xl, xu, f, g, A, frtol, fatol, fmin, cgtol, cg_itermax,
                &delta, &task, B, L, xc, s, indfree, gfree, isave, dsave, wa, iwa);
        }

        // [3] NEWX: a new point was computed.

        if (task == 3) {
            gnorm = gpnorm(n, x, xl, xu, g);
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

    free(g);
    free(xc);
    free(s);
    free(wa);
    free(gfree);
    free(dsave);
    free(indfree);
    free(iwa);
    free(isave);
    free(A);
    free(B);
    free(L);

    (*_status) = status;
    (*_minor_iter) = minor_iter;

    return;
}