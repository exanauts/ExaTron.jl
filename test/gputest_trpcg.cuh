__global__
void trpcg_kernel(int n, double *A, double *g, double *delta, double *L,
                  double tol, double stol, int itermax, double *w,
                  double *p, double *q, double *r, double *t, double *z,
                  int *info, int *iters)
{
    int bx = blockIdx.x;

    ctrpcg(n, &A[(n*n)*bx], &g[n*bx], delta[bx], &L[(n*n)*bx],
           tol, stol, itermax, &w[n*bx], &p[n*bx], &q[n*bx], &r[n*bx],
           &t[n*bx], &z[n*bx], &info[bx], &iters[bx]);
    return;
}

void test_trpcg(int n, int gridSize)
{
    int blockSize, totalMemSize;
    double tol = 1e-6, stol = 1e-6;
    int *info, *iters;
    double *A, *g, *delta, *L, *w, *p, *q, *r, *t, *z;

    printf("%-25s", "Testing trpcg()   . . .");
    fflush(stdout);

    std::mt19937 gen(0);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    blockSize = n*n;
    totalMemSize = sizeof(double)*gridSize*n;
    info = (int *)malloc(sizeof(int)*gridSize);
    iters = (int *)malloc(sizeof(int)*gridSize);
    L = (double *)malloc(sizeof(double)*gridSize*blockSize);
    A = (double *)malloc(sizeof(double)*gridSize*blockSize);
    g = (double *)malloc(totalMemSize);
    w = (double *)malloc(totalMemSize);
    p = (double *)malloc(totalMemSize);
    q = (double *)malloc(totalMemSize);
    r = (double *)malloc(totalMemSize);
    t = (double *)malloc(totalMemSize);
    z = (double *)malloc(totalMemSize);
    delta = (double *)malloc(sizeof(double)*gridSize);

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

    memcpy(A, L, sizeof(double)*gridSize*blockSize);

    for (int i = 0; i < gridSize; i++) {
        delta[i] = 100.0*dist(gen);
        for (int j = 0; j < n; j++) {
            g[n*i + j] = 0.1;
        }
    }

    int *devInfo, *devIters;
    double *devA, *devG, *devDelta, *devL, *devW, *devP, *devQ, *devR, *devT, *devZ;
    cudaMalloc(&devInfo, sizeof(int)*gridSize);
    cudaMalloc(&devIters, sizeof(int)*gridSize);
    cudaMalloc(&devL, sizeof(double)*gridSize*blockSize);
    cudaMalloc(&devA, sizeof(double)*gridSize*blockSize);
    cudaMalloc(&devG, totalMemSize);
    cudaMalloc(&devW, totalMemSize);
    cudaMalloc(&devP, totalMemSize);
    cudaMalloc(&devQ, totalMemSize);
    cudaMalloc(&devR, totalMemSize);
    cudaMalloc(&devT, totalMemSize);
    cudaMalloc(&devZ, totalMemSize);
    cudaMalloc(&devDelta, sizeof(double)*gridSize);
    cudaMemcpy(devL, L, sizeof(double)*gridSize*blockSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devA, A, sizeof(double)*gridSize*blockSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devG, g, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devDelta, delta, sizeof(double)*gridSize, cudaMemcpyHostToDevice);

    trpcg_kernel<<<gridSize, dim3(n,n)>>>(n, devA, devG, devDelta, devL, tol, stol, n,
                                          devW, devP, devQ, devR, devT, devZ, devInfo, devIters);
    cudaDeviceSynchronize();

    int *hostInfo = (int *)malloc(sizeof(int)*gridSize);
    int *hostIters = (int *)malloc(sizeof(int)*gridSize);
    double *hostW = (double *)malloc(totalMemSize);
    cudaMemcpy(hostInfo, devInfo, sizeof(int)*gridSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostIters, devIters, sizeof(int)*gridSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostW, devW, totalMemSize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < gridSize; i++) {
        trpcg(n, &A[(n*n)*i], &g[n*i], delta[i], &L[(n*n)*i], tol, stol, n,
              &w[n*i], &p[n*i], &q[n*i], &r[n*i], &t[n*i], &z[n*i], &info[i], &iters[i]);
    }

    double err = 0;
    for (int i = 0; i < gridSize; i++) {
        err = max(err, abs(hostW[i] - w[i]));
        err = max(err, (double)abs(hostInfo[i] - info[i]));
        err = max(err, (double)abs(hostIters[i] - iters[i]));
    }
    printf("%.5e\n", err);

    free(A);
    free(g);
    free(delta);
    free(L);
    free(w);
    free(p);
    free(q);
    free(r);
    free(t);
    free(z);
    free(info);
    free(iters);
    free(hostW);
    free(hostInfo);
    free(hostIters);
    cudaFree(devA);
    cudaFree(devG);
    cudaFree(devDelta);
    cudaFree(devL);
    cudaFree(devW);
    cudaFree(devP);
    cudaFree(devQ);
    cudaFree(devR);
    cudaFree(devT);
    cudaFree(devZ);
    cudaFree(devInfo);
    cudaFree(devIters);

    return;
}