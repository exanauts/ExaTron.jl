__device__
double cdot(int n, double *x, double *y)
{
    double v = 0;

    #pragma unroll
    for (int i = 0; i < n; i++) {
        v += x[i]*y[i];
    }
    __syncthreads();
    return v;
}