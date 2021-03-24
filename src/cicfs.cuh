__device__
void cicfs(int n, double alpha, double *A, double *L, double *wa1, double *wa2)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int nbmax = 3, nbfactor = 512;

    // Compute the l2 norms of the columns of A.
    cnrm2_mat(n, wa1, A);

    // Compute the scaling matrix D.
    if (tx < n && ty == 0) {
        wa2[tx] = (wa1[tx] > 0.0) ? 1.0/sqrt(wa1[tx]) : 1.0;
    }
    __syncthreads();

    // Determine a lower bound for the step.
    double alphas = (alpha <= 0.0) ? 1.0e-3 : alpha;

    // Compute the initial shift.
    alpha = 0.0;
    if (tx < n) {
        alpha = (A[tx*n + tx] == 0.0) ? alphas : max(alpha, -A[tx*n + tx]*(wa2[tx]*wa2[tx]));
    }
    __syncwarp();

    // Find the maximum alpha in a warp and put it in the first thread.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        alpha = max(alpha, __shfl_down_sync(0xffffffff, alpha, offset));
    }
    // Broadcast it to the entire threads in a warp.
    alpha = __shfl_sync(0xffffffff, alpha, 0);
    if (alpha > 0) {
        alpha = max(alpha, alphas);
    }

    int nb = 1, info = 0;
    while (1) {
        if (tx < n && ty == 0) {
            #pragma unroll
            for (int j = 0; j < n; j++) {
                L[n*tx + j] = A[n*tx + j]*(wa2[j]*wa2[tx]);
            }
            if (alpha != 0.0) {
                L[n*tx + tx] += alpha;
            }
        }
        /*
        if (tx < n && ty < n) {
            L[n*ty + tx] = A[n*ty + tx]*(wa2[ty]*wa2[tx]);
            if (alpha != 0.0 && tx == ty) {
                L[n*tx + tx] += alpha;
            }
        }
        */
        __syncthreads();

        // Attempt a Cholesky factorization.
        info = cicf(n, L);

        // If the factorization exists, then test for termination.
        // Otherwise, increment the shift.
        if (info >= 0) {
            // If the shift is at the lower bound, reduce the shift.
            // Otherwise, undo the scaling of L and exit.
            if ((alpha == alphas) && (nb < nbmax)) {
                alphas /= nbfactor;
                alpha = alphas;
                nb++;
            } else {
                if (tx < n && ty == 0) {
                    #pragma unroll
                    for (int j = 0; j < n; j++) {
                        if (tx >= j) {
                            L[n*j + tx] /= wa2[tx];
                            L[n*tx + j] = L[n*j + tx];
                        }
                    }
                }
                /*
                if (tx < n && ty < n && tx >= ty) {
                    L[n*ty + tx] /= wa2[tx];
                    L[n*tx + ty] = L[n*ty + tx];
                }
                */
                __syncthreads();
                return;
            }
        } else {
            alpha = max(2.0*alpha, alphas);
        }
    }

    return;
}
