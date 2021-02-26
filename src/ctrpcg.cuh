__device__
void ctrpcg(int n, double *A, double *g, double delta, double *L,
            double tol, double stol, int itermax, double *w,
            double *p, double *q, double *r, double *t, double *z,
            int *_info, int *_iters)
{
    // Initialize the iterate w and the residual r.
    for (int i = 0; i < n; i++) {
        w[i] = 0.0;
    }
    __syncthreads();

    // Initialize the residual t of grad q to -g.
    // Initialize the residual r of grad Q by solving L*r = -g.
    // Note that t = L*r.

    ccopy(n, g, t);
    cscal(n, -1.0, t);
    ccopy(n, t, r);
    cnsol(n, L, r);

    // Initialize the direction p.
    ccopy(n, r, p);

    //Initialize rho and the norms of r and t.
    double rho = cdot(n, r, r);
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

        ccopy(n, p, z);
        ctsol(n, L, z);

        // Compute q by solving L*q = A*z and save L*q for
        // use in updating the residual t.
        cssyax(n, A, z, q);
        ccopy(n, q, z);
        cnsol(n, L, q);

        // Compute alpha and determine sigma such that the trust region
        // constraint || w + sigma*p || = delta is satisfied.
        ptq = cdot(n, p, q);
        if (ptq > 0.0) {
            alpha = rho / ptq;
        } else {
            alpha = 0.0;
        }

        sigma = ctrqsol(n, w, p, delta);

        // Exit if there is negative curvature or if the
        // iterates exit the trust region.

        if (ptq <= 0.0 || alpha >= sigma) {
            caxpy(n, sigma, p, w);
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

        caxpy(n, alpha, p, w);
        caxpy(n, -alpha, q, r);
        caxpy(n, -alpha, z, t);

        // Exit if the residual convergence test is satisfied.

        rtr = cdot(n, r, r);
        rnorm = sqrt(rtr);
        tnorm = sqrt(cdot(n, t, t));

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
        cscal(n, beta, p);
        caxpy(n, 1.0, r, p);
        rho = rtr;
    }

    iters = itermax;
    (*_info) = 5;
    (*_iters) = iters;
    return;
}