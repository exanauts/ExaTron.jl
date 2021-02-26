__device__
void cprsrch(int n, double *x, double *xl, double *xu, double *A,
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
    cbreakpt(n, x, xl, xu, w, &nbrpt, &brptmin, &brptmax);

    bool search = true;
    while (search && alpha > brptmin) {

        // Calculate P[x + alpha*w] - x and check the sufficient
        // decrease condition.

        nsteps += 1;
        cgpstep(n, x, xl, xu, alpha, w, wa1);
        cssyax(n, A, wa1, wa2);
        gts = cdot(n, g, wa1);
        q = 0.5*cdot(n, wa1, wa2) + gts;
        if (q <= mu0*gts) {
            search = false;
        } else {

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

    cgpstep(n, x, xl, xu, alpha, w, wa1);
    caxpy(n, alpha, w, x);
    cmid(n, x, xl, xu);
    ccopy(n, wa1, w);

    return;
}