__device__
void cspcg(int n, double delta, double rtol, int itermax,
           double *x, double *xl, double *xu, double *A, double *g,
           double *s, double *B, double *L, int *indfree, double *gfree,
           double *w, int *iwa, double *wa, int *_info, int *_iters)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    double alpha, gfnorm, gfnormf, tol, stol;
    int info = 3, iters = 0, info_tr, iters_tr;
    int nfree;

    // Compute A*(x1 - x0) and store in w.
    cssyax(n, A, s, w);

    // Compute the Cauchy point.
    caxpy(n, 1.0, s, x);
    cmid(n, x, xl, xu);

    // Start the main iteration loop.
    // There are at most n ierations because at each iteration
    // at least one variable becomes active.

    for (int nfaces = 0; nfaces < n; nfaces++) {

        // Determine the free variables at the current minimizer.
        // The indices of the free variables are stored in the first
        // n free positions of the array indfree.
        // The array iwa is used to detect free variables by setting
        // iwa[i] = nfree if the ith variable is free, otherwise iwa[i] = 0.

        nfree = 0;
        if (tx == 0 && ty == 0) {
            for (int j = 0; j < n; j++) {
                if (xl[j] < x[j] && x[j] < xu[j]) {
                    indfree[nfree] = j;
                    iwa[j] = nfree;
                    nfree++;
                } else {
                    iwa[j] = 0;
                }
            }
            iwa[n] = nfree;
        }
        __syncthreads();
        nfree = iwa[n];
        __syncthreads();

        if (nfree == 0) {
            (*_info) = 1;
            (*_iters) = iters;
            return;
        }

        // Obtain the submatrix of A for the free variables.
        // Recall that iwa allows the detection of free variables.

        creorder(n, nfree, B, A, indfree, iwa);

        // Compute the Cholesky factorization with diagonal shifting.

        alpha = 0.0;
        cicfs(nfree, alpha, B, L, wa, &wa[n]);

        // Compute the gradient grad q(x[k]) = g + A*(x[k] - x[0])
        // of q at x[k] for the free variables.
        // Recall that w contains A*(x[k] - x[0]).
        // Compute the norm of the reduced gradient Z'*g.

        /*
        if (tx == 0 && ty == 0) {
            for (int j = 0; j < nfree; j++) {
                gfree[j] = w[indfree[j]] + g[indfree[j]];
                wa[j] = g[indfree[j]];
            }
        }
        */
        if (tx < nfree && ty == 0) {
            gfree[tx] = w[indfree[tx]] + g[indfree[tx]];
            wa[tx] = g[indfree[tx]];
        }
        __syncthreads();
        gfnorm = cnrm2(nfree, wa);

        // Save the trust region subproblem in the free variables
        // to generate a direction p[k]. Store p[k] in the array w.

        tol = rtol*gfnorm;
        stol = 0.0;

        ctrpcg(nfree, B, gfree, delta, L, tol, stol, itermax,
               w, wa, &wa[n], &wa[2*n], &wa[3*n], &wa[4*n],
               &info_tr, &iters_tr);
        iters += iters_tr;
        ctsol(nfree, L, w);

        // Use a projected search to obtain the next iterate.
        // The projected search algorithm stores s[k] in w.

        /*
        if (tx == 0 && ty == 0) {
            for (int j = 0; j < nfree; j++) {
                wa[j] = x[indfree[j]];
                wa[n + j] = xl[indfree[j]];
                wa[2*n + j] = xu[indfree[j]];
            }
        }
        */
        if (tx < nfree && ty == 0) {
            wa[tx] = x[indfree[tx]];
            wa[n + tx] = xl[indfree[tx]];
            wa[2*n + tx] = xu[indfree[tx]];
        }
        __syncthreads();

        cprsrch(nfree, wa, &wa[n], &wa[2*n], B, gfree, w, &wa[3*n], &wa[4*n]);

        // Update the minimizer and the step.
        // Note that s now contains x[k+1] - x[0].

        /*
        if (tx == 0 && ty == 0) {
            for (int j = 0; j < nfree; j++) {
                x[indfree[j]] = wa[j];
                s[indfree[j]] += w[j];
            }
        }
        */
        if (tx < nfree && ty == 0) {
            x[indfree[tx]] = wa[tx];
            s[indfree[tx]] += w[tx];
        }
        __syncthreads();

        // Compute A*(x[k+1] - x[0]) and store in w.
        cssyax(n, A, s, w);

        // Compute the gradient grad q(x[k+1]) = g + A*(x[k+1] - x[0])
        // of q at x[k+1] for the free variables.

        if (tx == 0 && ty == 0) {
            #pragma unroll
            for (int j = 0; j < nfree; j++) {
                gfree[j] = w[indfree[j]] + g[indfree[j]];
            }
        }
        __syncthreads();

        gfnormf = cnrm2(nfree, gfree);

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