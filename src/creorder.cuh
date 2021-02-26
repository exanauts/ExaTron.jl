__device__
void creorder(int n, int nfree, double *B, double *A,
              int *indfree, int *iwa)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int jfree;
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
    __syncthreads();

    return;
}