__device__
void cssyax(int n, double *A, double *z, double *q)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    double v = 0.0;
    if (tx < n && ty < n) {
        v = A[tx*n + ty]*z[tx];
    }

    //Sum over the x-dimension: v = sum_tx A[ty,tx]*z[tx].
    //The thread with tx=0 will have the sum in v.

    int offset = blockDim.x / 2;
    while (offset > 0) {
        v += __shfl_down_sync(0xffffffff, v, offset);
        offset >>= 1;
    }

    if (tx == 0) {
        q[ty] = v;
    }
    __syncthreads();

    return;
}