__global__
void cauchy_kernel(int n, double *x, double *xl, double *xu,
                   double *A, double *g, double *delta, double *alpha,
                   double *s, double *wa)
{
    int bx = blockIdx.x;

    alpha[bx] = ccauchy(n, &x[n*bx], &xl[n*bx], &xu[n*bx], &A[(n*n)*bx], &g[n*bx],
                        delta[bx], alpha[bx], &s[n*bx], &wa[n*bx]);
    return;
}

void test_cauchy(int n, int gridSize)
{
    int blockSize, totalMemSize;
    double *x, *xl, *xu, *L, *A, *g, *delta, *alpha, *s, *wa;

    printf("%-25s", "Testing cauchy()  . . .");
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
    delta = (double *)malloc(sizeof(double)*gridSize);
    alpha = (double *)malloc(sizeof(double)*gridSize);
    s = (double *)malloc(totalMemSize);
    wa = (double *)malloc(totalMemSize);

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
        alpha[i] = dist(gen);
        delta[i]= 2*nrm2_vec(n, &g[n*i]);
    }

    for (int i = 0; i < gridSize; i++) {
        alpha[i] = cauchy(n, &x[n*i], &xl[n*i], &xu[n*i], &A[blockSize*i], &g[n*i],
                          delta[i], alpha[i], &s[n*i], &wa[n*i]);
    }

    double *devX, *devXl, *devXu, *devA, *devG, *devDelta, *devAlpha, *devS, *devWa;

    cudaMalloc(&devX, totalMemSize);
    cudaMalloc(&devXl, totalMemSize);
    cudaMalloc(&devXu, totalMemSize);
    cudaMalloc(&devA, sizeof(double)*gridSize*blockSize);
    cudaMalloc(&devG, totalMemSize);
    cudaMalloc(&devDelta, sizeof(double)*gridSize);
    cudaMalloc(&devAlpha, sizeof(double)*gridSize);
    cudaMalloc(&devS, totalMemSize);
    cudaMalloc(&devWa, totalMemSize);
    cudaMemcpy(devX, x, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devXl, xl, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devXu, xu, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devA, A, sizeof(double)*gridSize*blockSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devG, g, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devAlpha, alpha, sizeof(double)*gridSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devDelta, delta, sizeof(double)*gridSize, cudaMemcpyHostToDevice);

    cauchy_kernel<<<gridSize, dim3(n,n)>>>(n, devX, devXl, devXu, devA, devG,
                                           devDelta, devAlpha, devS, devWa);
    cudaDeviceSynchronize();

    double *hostAlpha = (double *)malloc(sizeof(double)*gridSize);
    cudaMemcpy(hostAlpha, devAlpha, sizeof(double)*gridSize, cudaMemcpyDeviceToHost);

    double err = 0;
    for (int i = 0; i < gridSize; i++) {
        err = max(err, abs(hostAlpha[i] - alpha[i]));
    }
    printf("%.5e\n", err);

    free(x);
    free(xl);
    free(xu);
    free(L);
    free(A);
    free(g);
    free(delta);
    free(alpha);
    free(s);
    free(wa);
    cudaFree(devX);
    cudaFree(devXl);
    cudaFree(devXu);
    cudaFree(devA);
    cudaFree(devG);
    cudaFree(devDelta);
    cudaFree(devAlpha);
    cudaFree(devS);
    cudaFree(devWa);

    return;
}