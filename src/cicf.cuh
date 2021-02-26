__device__
int cicf(int n, double *L)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    double Lmn, Ljj;

    Lmn = L[n*ty + tx];
    for (int j = 0; j < n; j++) {
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

    if (tx < n && ty < n && tx > ty) {
        L[n*tx + ty] = L[n*ty + tx];
    }
    __syncthreads();

    return 0;
}
