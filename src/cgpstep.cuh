__device__
void cgpstep(int n, double *x, double *xl, double *xu,
             double alpha, double *w, double *s)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (tx < n && ty == 0) {
        if (x[tx] + alpha*w[tx] < xl[tx]) {
            s[tx] = xl[tx] - x[tx];
        } else if (x[tx] + alpha*w[tx] > xu[tx]) {
            s[tx] = xu[tx] - x[tx];
        } else {
            s[tx] = alpha*w[tx];
        }
    }
    __syncthreads();

    return;
}