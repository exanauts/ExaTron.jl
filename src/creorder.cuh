__device__
void creorder(int n, int nfree, double * __restrict__ B, const double * __restrict__ A,
              const int * __restrict__ indfree, const int * __restrict__ iwa)
//void creorder(int n, int nfree, double *B, double *A,
//              int *indfree, int *iwa)

{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int jfree;

    /*
    if (tx == 0 && ty == 0) {
        for (int j = 0; j < nfree; j++) {
            jfree = indfree[j];
            B[nfree*j + j] = A[n*jfree + jfree];
            for (int i = jfree+1; i < n; i++) {
                if (iwa[i] > 0) {
                    B[nfree*j + iwa[i]] = A[n*jfree + i];
                    B[nfree*iwa[i] + j] = B[nfree*j + iwa[i]];
                }
            }
        }
    }
    */
    if (tx < nfree && ty == 0){
        jfree = indfree[tx];
        B[nfree*tx + tx] = A[n*jfree + jfree];
        #pragma unroll
        for (int i = jfree+1; i < n; i++) {
            if (iwa[i] > 0) {
                B[nfree*tx + iwa[i]] = A[n*jfree + i];
                B[nfree*iwa[i] + tx] = B[nfree*tx + iwa[i]];
            }
        }
    }
    __syncthreads();
    return;
}