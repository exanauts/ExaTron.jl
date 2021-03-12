__device__
void cnsol(int n, double *L, double *r)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Solve L*x = r and store the result in r.
    #pragma unroll
    for (int j = 0; j < n; j++) {
        if (tx == 0 && ty == 0) {
            r[j] /= L[n*j + j];
        }
        __syncthreads();

        if (tx > j && tx < n && ty == 0) {
            r[tx] -= L[n*j + tx]*r[j];
        }
        __syncthreads();
    }

    return;
}

__device__
void ctsol(int n, double *L, double *r)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Solve L'* = r and store the result in r.
    #pragma unroll
    for (int j = n-1; j >= 0; j--) {
        if (tx == 0 && ty == 0) {
            r[j] /= L[n*j + j];
        }
        __syncthreads();

        if (tx < j && ty == 0) {
            r[tx] -= L[n*j + tx]*r[j];
        }
        __syncthreads();
    }

    return;
}