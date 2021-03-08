__device__
void cnrm2_mat(int n, double * __restrict__ wa, const double * __restrict__ A)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //double v = A[n*ty + tx]*A[n*ty + tx];
    double v = A[blockDim.x*ty + tx]*A[blockDim.x*ty + tx];
    if (tx >= n || ty >= n) {
        v = 0.0;
    }

    int offset = blockDim.x / 2;
    while (offset > 0) {
        v += __shfl_down_sync(0xffffffff, v, offset);
        offset >>= 1;
    }

    if (tx == 0) {
        wa[ty] = sqrt(v);
    }
    __syncthreads();

    return;
}

__device__
double cnrm2(int n, const double * __restrict__ x)
{
    int tx = threadIdx.x;

    double v = 0.0;
    if (tx < n) {
        v = x[tx]*x[tx];
    }
    __syncthreads();

    int offset = blockDim.x / 2;
    while (offset > 0) {
        v += __shfl_down_sync(0xffffffff, v, offset);
        offset >>= 1;
    }

    v = sqrt(v);
    v = __shfl_sync(0xffffffff, v, 1);
    return v;
}