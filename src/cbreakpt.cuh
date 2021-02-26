__device__
void cbreakpt(int n, double *x, double *xl, double *xu, double *w,
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
    __syncthreads();

    return;
}