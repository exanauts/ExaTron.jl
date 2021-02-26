__global__
void tron_kernel(int n, double *x, double *xl, double *xu, double *f, double *g,
                 double *A, double frtol, double fatol, double fmin, double cgtol,
                 int itermax, double *delta, int *task, double *B, double *L,
                 double *xc, double *s, int *indfree, double *gfree, int *isave,
                 double *dsave, double *wa, int *iwa)
{
    int bx = blockIdx.x;

    ctron(n, &x[n*bx], &xl[n*bx], &xu[n*bx], f[bx], &g[n*bx], &A[(n*n)*bx],
          frtol, fatol, fmin, cgtol, itermax, &delta[bx], &task[bx],
          &B[(n*n)*bx], &L[(n*n)*bx], &xc[n*bx], &s[n*bx], &indfree[n*bx],
          &gfree[n*bx], &isave[3*bx], &dsave[3*bx], &wa[7*n*bx], &iwa[3*n*bx]);
    return;
}

void test_tron(int n, int gridSize)
{
    int blockSize, totalMemSize;
    double *x, *xl, *xu, *xc, *L, *A, *B, *f, *g, *s, *gfree, *wa, *delta, *dsave;
    int *indfree, *iwa, *isave, *task;

    printf("%-25s", "Testing tron()    . . .");
    fflush(stdout);

    std::mt19937 gen(0);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    blockSize = n*n;
    totalMemSize = sizeof(double)*gridSize*n;
    x = (double *)malloc(totalMemSize);
    xl = (double *)malloc(totalMemSize);
    xu = (double *)malloc(totalMemSize);
    xc = (double *)malloc(totalMemSize);
    L = (double *)malloc(sizeof(double)*gridSize*blockSize);
    A = (double *)malloc(sizeof(double)*gridSize*blockSize);
    B = (double *)malloc(sizeof(double)*gridSize*blockSize);
    f = (double *)malloc(sizeof(double)*gridSize);
    g = (double *)malloc(totalMemSize);
    s = (double *)malloc(totalMemSize);
    gfree = (double *)malloc(totalMemSize);
    wa = (double *)malloc(7*totalMemSize);
    delta = (double *)malloc(sizeof(double)*gridSize);
    dsave = (double *)malloc(sizeof(double)*gridSize*(3));
    indfree = (int *)malloc(sizeof(int)*gridSize*n);
    iwa = (int *)malloc(sizeof(int)*gridSize*(3*n));
    isave = (int *)malloc(sizeof(int)*gridSize*(3));
    task = (int *)malloc(sizeof(int)*gridSize);

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
        f[i] = 0.5*dot(n, &g[n*i], &x[n*i]); // f = 0.5*x'*A*x
        for (int j = 0; j < n; j++) {
            double c = dist(gen);
            f[i] += c*x[n*i + j]; // f += c'*x
            g[n*i + j] += c;
            s[n*i + j] = dist(gen);
        }
        delta[i] = 2.0*nrm2_vec(n, &g[n*i]);
    }

    memset(wa, 0, 7*totalMemSize);
    memset(gfree, 0, totalMemSize);
    memset(indfree, 0, sizeof(int)*gridSize*n);
    memset(iwa, 0, sizeof(int)*gridSize*(3*n));
    memset(task, 0, sizeof(int)*gridSize);

    double *devX, *devXl, *devXu, *devXc, *devL, *devA, *devB, *devF, *devG, *devS, *devGfree, *devWa, *devDelta, *devDsave;
    int *devIndfree, *devIwa, *devIsave, *devTask;

    cudaMalloc(&devX, totalMemSize);
    cudaMalloc(&devXl, totalMemSize);
    cudaMalloc(&devXu, totalMemSize);
    cudaMalloc(&devXc, totalMemSize);
    cudaMalloc(&devL, sizeof(double)*gridSize*blockSize);
    cudaMalloc(&devA, sizeof(double)*gridSize*blockSize);
    cudaMalloc(&devB, sizeof(double)*gridSize*blockSize);
    cudaMalloc(&devF, sizeof(double)*gridSize);
    cudaMalloc(&devG, totalMemSize);
    cudaMalloc(&devS, totalMemSize);
    cudaMalloc(&devGfree, totalMemSize);
    cudaMalloc(&devWa, 7*totalMemSize);
    cudaMalloc(&devDelta, sizeof(double)*gridSize);
    cudaMalloc(&devDsave, sizeof(double)*gridSize*3);
    cudaMalloc(&devIndfree, sizeof(int)*gridSize*n);
    cudaMalloc(&devIwa, sizeof(int)*gridSize*(3*n));
    cudaMalloc(&devIsave, sizeof(int)*gridSize*3);
    cudaMalloc(&devTask, sizeof(int)*gridSize);
    cudaMemcpy(devX, x, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devXl, xl, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devXu, xu, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devA, A, sizeof(double)*gridSize*blockSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devF, f, sizeof(double)*gridSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devG, g, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devS, s, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devDelta, delta, sizeof(double)*gridSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devTask, task, sizeof(int)*gridSize, cudaMemcpyHostToDevice);

    int cg_itermax = n;
    double fatol = 0.0, frtol = 1e-12, fmin = -1e-32, cgtol = 0.1;

    tron_kernel<<<gridSize, dim3(n,n)>>>(n, devX, devXl, devXu, devF, devG, devA,
                                         frtol, fatol, fmin, cgtol, cg_itermax, devDelta,
                                         devTask, devB, devL, devXc, devS, devIndfree,
                                         devGfree, devIsave, devDsave, devWa, devIwa);
    cudaDeviceSynchronize();

    double *hostX = (double *)malloc(totalMemSize);
    cudaMemcpy(hostX, devX, totalMemSize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < gridSize; i++) {
        tron(n, &x[n*i], &xl[n*i], &xu[n*i], f[i], &g[n*i], &A[(n*n)*i],
             frtol, fatol, fmin, cgtol, cg_itermax, &delta[i], &task[i],
             &B[(n*n)*i], &L[(n*n)*i], &xc[n*i], &s[n*i], &indfree[n*i],
             &gfree[n*i], &isave[3*i], &dsave[3*i], &wa[7*n*i], &iwa[3*n*i]);
    }

    double err = 0;
    for (int i = 0; i < gridSize; i++) {
        err = max(err, diff_vector(n, &hostX[n*i], &x[n*i]));
    }
    printf("%.5e\n", err);

    free(x);
    free(xl);
    free(xu);
    free(xc);
    free(L);
    free(A);
    free(B);
    free(g);
    free(s);
    free(gfree);
    free(wa);
    free(delta);
    free(indfree);
    free(iwa);
    free(hostX);
    cudaFree(devX);
    cudaFree(devXl);
    cudaFree(devXu);
    cudaFree(devXc);
    cudaFree(devL);
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devG);
    cudaFree(devS);
    cudaFree(devGfree);
    cudaFree(devWa);
    cudaFree(devDelta);
    cudaFree(devIndfree);
    cudaFree(devIwa);

    return;
}