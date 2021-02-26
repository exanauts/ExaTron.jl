__global__
void spcg_kernel(int n, double *delta, double rtol, int itermax,
                 double *x, double *xl, double *xu, double *A, double *g,
                 double *s, double *B, double *L, int *indfree, double *gfree,
                 double *w, int *iwa, double *wa, int *info, int *iters)
{
    int bx = blockIdx.x;

    cspcg(n, delta[bx], rtol, itermax, &x[n*bx], &xl[n*bx], &xu[n*bx],
          &A[(n*n)*bx], &g[n*bx], &s[n*bx], &B[(n*n)*bx], &L[(n*n)*bx],
          &indfree[n*bx], &gfree[n*bx], &w[n*bx], &iwa[(2*n)*bx], &wa[(5*n)*bx],
          &info[bx], &iters[bx]);
    return;
}

void test_spcg(int n, int gridSize)
{
    int blockSize, totalMemSize;
    double *x, *xl, *xu, *L, *A, *B, *g, *s, *gfree, *w, *wa, *delta;
    int *indfree, *iwa, *info, *iters;

    printf("%-25s", "Testing spcg()    . . .");
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
    B = (double *)malloc(sizeof(double)*gridSize*blockSize);
    g = (double *)malloc(totalMemSize);
    s = (double *)malloc(totalMemSize);
    gfree = (double *)malloc(totalMemSize);
    w = (double *)malloc(totalMemSize);
    wa = (double *)malloc(5*totalMemSize);
    delta = (double *)malloc(sizeof(double)*gridSize);
    indfree = (int *)malloc(sizeof(int)*gridSize*n);
    iwa = (int *)malloc(sizeof(int)*gridSize*(2*n));
    info = (int *)malloc(sizeof(int)*gridSize);
    iters = (int *)malloc(sizeof(int)*gridSize);

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
    for (int i = 0; i < gridSize; i++)
        llt(n, &A[blockSize*i], &L[blockSize*i]);

    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < n; j++) {
            x[n*i + j] = dist(gen);
            xl[n*i + j] = x[n*i + j] - dist(gen);
            xu[n*i + j] = x[n*i + j] + dist(gen);
        }
        gemv(n, &g[n*i], 1.0, &A[blockSize*i], &x[n*i], 0.0, NULL);
        for (int j = 0; j < n; j++) {
            g[n*i + j] += dist(gen);
            s[n*i + j] = dist(gen);
        }
        delta[i] = 2.0*nrm2_vec(n, &g[n*i]);
    }
    memset(w, 0, totalMemSize);
    memset(wa, 0, 5*totalMemSize);
    memset(gfree, 0, totalMemSize);
    memset(indfree, 0, sizeof(int)*gridSize*n);
    memset(iwa, 0, sizeof(int)*gridSize*(2*n));

    double *devX, *devXl, *devXu, *devL, *devA, *devB, *devG, *devS, *devGfree, *devW, *devWa, *devDelta;
    int *devIndfree, *devIwa, *devInfo, *devIters;

    cudaMalloc(&devX, totalMemSize);
    cudaMalloc(&devXl, totalMemSize);
    cudaMalloc(&devXu, totalMemSize);
    cudaMalloc(&devL, sizeof(double)*gridSize*blockSize);
    cudaMalloc(&devA, sizeof(double)*gridSize*blockSize);
    cudaMalloc(&devB, sizeof(double)*gridSize*blockSize);
    cudaMalloc(&devG, totalMemSize);
    cudaMalloc(&devS, totalMemSize);
    cudaMalloc(&devGfree, totalMemSize);
    cudaMalloc(&devW, totalMemSize);
    cudaMalloc(&devWa, 5*totalMemSize);
    cudaMalloc(&devDelta, sizeof(double)*gridSize);
    cudaMalloc(&devIndfree, sizeof(int)*gridSize*n);
    cudaMalloc(&devIwa, sizeof(int)*gridSize*(2*n));
    cudaMalloc(&devInfo, sizeof(int)*gridSize);
    cudaMalloc(&devIters, sizeof(int)*gridSize);
    cudaMemcpy(devX, x, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devXl, xl, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devXu, xu, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devA, A, sizeof(double)*gridSize*blockSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devG, g, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devS, s, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devDelta, delta, sizeof(double)*gridSize, cudaMemcpyHostToDevice);

    double rtol = 1e-6;

    spcg_kernel<<<gridSize, dim3(n,n)>>>(n, devDelta, rtol, n, devX, devXl, devXu, devA, devG,
                                         devS, devB, devL, devIndfree, devGfree, devW, devIwa, devWa,
                                         devInfo, devIters);
    cudaDeviceSynchronize();

    double *hostX = (double *)malloc(totalMemSize);
    cudaMemcpy(hostX, devX, totalMemSize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < gridSize; i++) {
        spcg(n, delta[i], rtol, n, &x[n*i], &xl[n*i], &xu[n*i],
             &A[(n*n)*i], &g[n*i], &s[n*i], &B[(n*n)*i], &L[(n*n)*i],
             &indfree[n*i], &gfree[n*i], &w[n*i], &iwa[(2*n)*i], &wa[(5*n)*i],
             &info[i], &iters[i]);
    }

    double err = 0;
    for (int i = 0; i < gridSize; i++) {
        err = max(err, diff_vector(n, &hostX[n*i], &x[n*i]));
    }
    printf("%.5e\n", err);

    free(x);
    free(xl);
    free(xu);
    free(L);
    free(A);
    free(B);
    free(g);
    free(s);
    free(gfree);
    free(w);
    free(wa);
    free(delta);
    free(indfree);
    free(iwa);
    free(info);
    free(iters);
    free(hostX);
    cudaFree(devX);
    cudaFree(devXl);
    cudaFree(devXu);
    cudaFree(devL);
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devG);
    cudaFree(devS);
    cudaFree(devGfree);
    cudaFree(devW);
    cudaFree(devWa);
    cudaFree(devDelta);
    cudaFree(devIndfree);
    cudaFree(devIwa);
    cudaFree(devInfo);
    cudaFree(devIters);

    return;
}