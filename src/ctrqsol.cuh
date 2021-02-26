__device__
double ctrqsol(int n, double *x, double *p, double delta)
{
    double sigma = 0.0;

    double ptx = cdot(n, p, x);
    double ptp = cdot(n, p, p);
    double xtx = cdot(n, x, x);
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
    __syncthreads();

    return sigma;
}