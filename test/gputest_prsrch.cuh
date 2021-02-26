__global__
void prsrch_kernel(int n, double *x, double *xl, double *xu, double *A,
                   double *g, double *w, double *wa1, double *wa2)
{
    int bx = blockIdx.x;

    cprsrch(n, &x[n*bx], &xl[n*bx], &xu[n*bx], &A[(n*n)*bx],
            &g[n*bx], &w[n*bx], &wa1[n*bx], &wa2[n*bx]);
    return;
}

void test_prsrch(int n, int gridSize)
{
    int blockSize, totalMemSize;
    double *x, *xl, *xu, *L, *A, *g, *w, *wa1, *wa2;

    printf("%-25s", "Testing prsrch()  . . .");
    fflush(stdout);

    std::mt19937 gen(0);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    blockSize = n*n;
    totalMemSize = sizeof(double)*gridSize*n;
    x = (double *)malloc(totalMemSize);
    xl = (double *)malloc(totalMemSize);
    xu = (double *)malloc(totalMemSize);
    L = (double *)malloc(sizeof(double)*gridSize*blockSize);
    A = (double *)malloc(sizeof(double)*gridSize*blockSize);
    g = (double *)malloc(totalMemSize);
    w = (double *)malloc(totalMemSize);
    wa1 = (double *)malloc(totalMemSize);
    wa2 = (double *)malloc(totalMemSize);

    memset(L, 0, sizeof(double)*gridSize*blockSize);
    for (int i = 0; i < gridSize; i++) {
        double *Li = &L[blockSize*i];
        for (int j = 0; j < n; j++) {
            for (int k = j; k < n; k++) {
                Li[j*n + k] = 1e-1 + dist(gen);
            }
        }
    }

    memset(A, 0, sizeof(double)*gridSize*blockSize);
    for (int i = 0; i < gridSize; i++) {
        llt(n, &A[blockSize*i], &L[blockSize*i]);
    }

    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < n; j++) {
            x[n*i + j] = dist(gen);
            xl[n*i + j] = x[n*i + j] - dist(gen);
            xu[n*i + j] = x[n*i + j] + dist(gen);
        }
        gemv(n, &g[n*i], 1.0, &A[blockSize*i], &x[n*i], 0.0, NULL);
        for (int j = 0; j < n; j++) {
            g[n*i + j] += dist(gen);
        }
        for (int j = 0; j < n; j++) {
            w[n*i + j] = -g[n*i + j];
        }
    }

    double *devX, *devXl, *devXu, *devA, *devG, *devW, *devWa1, *devWa2;
    cudaMalloc(&devX, totalMemSize);
    cudaMalloc(&devXl, totalMemSize);
    cudaMalloc(&devXu, totalMemSize);
    cudaMalloc(&devA, sizeof(double)*gridSize*blockSize);
    cudaMalloc(&devG, totalMemSize);
    cudaMalloc(&devW, totalMemSize);
    cudaMalloc(&devWa1, totalMemSize);
    cudaMalloc(&devWa2, totalMemSize);
    cudaMemcpy(devX, x, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devXl, xl, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devXu, xu, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devA, A, sizeof(double)*gridSize*blockSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devG, g, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devW, w, totalMemSize, cudaMemcpyHostToDevice);

    prsrch_kernel<<<gridSize, dim3(n,n)>>>(n, devX, devXl, devXu, devA, devG, devW, devWa1, devWa2);
    cudaDeviceSynchronize();

    double *hostX, *hostW;
    hostX = (double *)malloc(totalMemSize);
    hostW = (double *)malloc(totalMemSize);
    cudaMemcpy(hostX, devX, totalMemSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostW, devW, totalMemSize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < gridSize; i++) {
        prsrch(n, &x[n*i], &xl[n*i], &xu[n*i], &A[(n*n)*i], &g[n*i], &w[n*i], &wa1[n*i], &wa2[n*i]);
    }

    double err = 0;
    for (int i = 0; i < gridSize; i++) {
        err = max(err, diff_vector(n, &hostX[n*i], &x[n*i]));
        err = max(err, diff_vector(n, &hostW[n*i], &w[n*i]));
    }
    printf("%.5e\n", err);

    free(x);
    free(xl);
    free(xu);
    free(A);
    free(g);
    free(w);
    free(wa1);
    free(wa2);
    cudaFree(devX);
    cudaFree(devXl);
    cudaFree(devXu);
    cudaFree(devA);
    cudaFree(devG);
    cudaFree(devW);
    cudaFree(devWa1);
    cudaFree(devWa2);

    return;
}