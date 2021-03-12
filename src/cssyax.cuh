__device__
void cssyax(int n, const double * __restrict__ A,
            const double * __restrict__ z, double * __restrict__ q)
{
#if 0
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    double v = 0.0;

    // Coalesced access but slower than #else code.
    if (tid < n) {
        #pragma unroll
        for (int j = 0; j < n; j++) {
            v += A[n*j + tid] * z[j];
        }
        q[tid] = v;
    }
    __syncthreads();
#else

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    double v = 0.0;

    if (tx < n && ty == 0) {
        #pragma unroll
        for (int j = 0; j < n; j++) {
            v += A[n*j + tx]*z[j];
        }
        q[tx] = v;
    }
/*
    if (tx < n && ty < n) {
        v = A[n*tx + ty]*z[tx];
    }

    //Sum over the x-dimension: v = sum_tx A[ty,tx]*z[tx].
    //The thread with tx=0 will have the sum in v.

    #pragma unroll
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }

    if (tx == 0) {
        q[ty] = v;
    }
*/
    __syncthreads();

#endif
    return;
}
