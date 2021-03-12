#if 0
__device__
int cicf(int n, double *L)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    double Lmn, Ljj;

    Lmn = L[n*ty + tx];
    #pragma unroll
    for (int j = 0; j < n; j++) {
        // __syncthreads();  // false positive from cuda-memcheck

        // Update the diagonal.
        if (tx == j && ty == j) {
            if (Lmn > 0) {
                L[n*j + j] = sqrt(Lmn);
            } else {
                L[n*j + j] = -1.0;
            }
        }
        __syncthreads();

        Ljj = L[n*j + j];
        if (Ljj <= 0) {
            __syncthreads();
            return -1;
        }

        // Update the jth column.
        if (ty == j && tx > j && tx < n) {
            L[n*j + tx] = Lmn / Ljj;
        }
        __syncthreads();

        // Update the trailing submatrix.
        Lmn -= L[n*j + tx] * L[n*j + ty];
        //Lmn -= __dmul_rn(L[n*j + tx], L[n*j + ty]);
    }

    // __syncthreads(); // false positive from cuda-memcheck

    if (tx < n && ty < n && tx > ty) {
        L[n*tx + ty] = L[n*ty + tx];
    }
    __syncthreads();

    return 0;
}
#else
__device__
int cicf(int n, double *L)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    double Ljj;

    #pragma unroll
    for (int j = 0; j < n; j++) {
        if (j > 0) {
            if (tx >= j && tx < n && ty == 0) {
                for (int k = 0; k < j; k++) {
                    L[n*j + tx] -= L[n*k + tx] * L[n*k + j];
                }
            }
        }
        __syncthreads();

        if (L[n*j + j] <= 0) {
            __syncthreads();
            return -1;
        }

        Ljj = sqrt(L[n*j + j]);
        if (tx >= j && tx < n && ty == 0) {
            L[n*j + tx] /= Ljj;
        }
        __syncthreads();
    }

    // __syncthreads(); // false positive from cuda-memcheck

    if (tx < n && ty == 0) {
        #pragma unroll
        for (int j = 0; j < n; j++) {
            if (tx > j) {
                L[n*tx + j] = L[n*j + tx];
            }
        }
    }
    /*
    if (tx < n && ty < n && tx > ty) {
        L[n*tx + ty] = L[n*ty + tx];
    }
    */
    __syncthreads();

    return 0;
}
#endif