__device__
void caxpy(int n, double a, double *x, double *y)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (ty == 0) {
        y[tx] += a*x[tx];
    }
    __syncthreads();

    return;
}