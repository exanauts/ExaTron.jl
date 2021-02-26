__device__
double ccauchy(int n, double *x, double *xl, double *xu,
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
    ccopy(n, g, wa);
    cscal(n, -1.0, wa);

    int nbrpt;
    double brptmin, brptmax;
    cbreakpt(n, x, xl, xu, wa, &nbrpt, &brptmin, &brptmax);

    // Evaluate the initial alpha and decide if the algorithm
    // must interpolate or extrapolate.

    bool interp;
    cgpstep(n, x, xl, xu, -alpha, g, s);
    if (cnrm2(n, s) > delta) {
        interp = true;
    } else {
        cssyax(n, A, s, wa);
        gts = cdot(n, g, s);
        q = 0.5*cdot(n, s, wa) + gts;
        interp = (q >= mu0*gts);
    }

    // Either interpolate or extrapolate to find a successful step.

    bool search = true;
    if (interp) {
        while (search) {
            alpha *= interpf;
            cgpstep(n, x, xl, xu, -alpha, g, s);
            if (cnrm2(n, s) <= delta) {
                cssyax(n, A, s, wa);
                gts = cdot(n, g, s);
                q = 0.5*cdot(n, s, wa) + gts;
                search = (q > mu0*gts);
            }
        }
    } else {
        // Increase alpha until a successful step is found.

        alphas = alpha;
        while (search && alpha <= brptmax) {
            alpha *= extrapf;
            cgpstep(n, x, xl, xu, -alpha, g, s);
            if (cnrm2(n, s) <= delta) {
                cssyax(n, A, s, wa);
                gts = cdot(n, g, s);
                q = 0.5*cdot(n, s, wa) + gts;
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
        cgpstep(n, x, xl, xu, -alpha, g, s);
    }

    return alpha;
}