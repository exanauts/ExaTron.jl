__device__
double cgpnorm(int n, double *x, double *xl, double *xu, double *g)
{
    int tx = threadIdx.x;

    double v = 0;
    if (tx < n) {
        if (xl[tx] != xu[tx]) {
            if (x[tx] == xl[tx]) {
                v = min(g[tx], 0.0);
            } else if (x[tx] == xu[tx]) {
                v = max(g[tx], 0.0);
            } else {
                v = g[tx];
            }

            v = abs(v);
        }
    }
    __syncthreads();

    #pragma unroll
    for (int offset = (n-1)/2 + 1; offset > 0; offset >>= 1) {
        v = max(v, __shfl_down_sync(0xffffffff, v, offset));
    }
    v = __shfl_sync(0xffffffff, v, 0);
    return v;
}
