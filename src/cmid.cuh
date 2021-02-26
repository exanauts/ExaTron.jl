__device__
void cmid(int n, double *x, double *xl, double *xu)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (ty == 0) {
        x[tx] = max(xl[tx], min(xu[tx], x[tx]));
    }
    __syncthreads();
    return;
}