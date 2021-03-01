__device__
void ctron(int n, double *x, double *xl, double *xu, double f, double *g,
           double *A, double frtol, double fatol, double fmin, double cgtol,
           int itermax, double *_delta, int *_task, double *B, double *L,
           double *xc, double *s, int *indfree, double *gfree, int *isave,
           double *dsave, double *wa, int *iwa)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

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

    __syncthreads();

    // Search for a lower function value.

    search = true;
    while (search) {

        // Compute a step and evaluate the function at the trial point.

        if (work == 1) { // "COMPUTE"

            // Save the best function value, iterate, and gradient.

            fc = f;
            ccopy(n, x, xc);

            // Compute the Cauchy step and store in s.

            alphac = ccauchy(n, x, xl, xu, A, g, delta, alphac, s, wa);

            // Compute the projected Newton step.

            cspcg(n, delta, cgtol, itermax, x, xl, xu, A, g, s,
                  B, L, indfree, gfree, wa, iwa, &wa[n], &info, &iters);

            // Compute the predicted reduction.

            cssyax(n, A, s, wa);
            prered = -(cdot(n, s, g) + 0.5*cdot(n, s, wa));
            iterscg = iterscg + iters;

            // Set task to compute the function.

            task = 1; // 'F'
        }

        // Evaluate the step and determine if the step is successful.

        if (work == 2) { // "EVALUATE"

            // Compute the actual reduction.

            actred = fc - f;

            // On the first iteration, adjust the initial step bound.

            snorm = cnrm2(n, s);
            if (iter == 1) {
                delta = min(delta, snorm);
            }

            // Update the trust region bound.

            g0 = cdot(n, g, s);
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
                ccopy(n, xc, x);
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

    if (tx == 0 && ty == 0) {
        isave[0] = work;
        isave[1] = iter;
        isave[2] = iterscg;

        dsave[0] = fc;
        dsave[1] = alphac;
        dsave[2] = prered;
    }

    (*_delta) = delta;
    (*_task) = task;

    __syncthreads();

    return;
}