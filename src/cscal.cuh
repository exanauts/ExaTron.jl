__device__
void cscal(int n, double s, double *x)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (ty == 0) {
        x[tx] *= s;
    }
    __syncthreads();
    return;
}