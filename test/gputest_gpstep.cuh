__global__
void gpstep_kernel(int n, double *x, double *xl, double *xu,
                   double *alpha, double *w, double *s)
{
    int bx = blockIdx.x;

    cgpstep(n, &x[n*bx], &xl[n*bx], &xu[n*bx], alpha[bx], &w[n*bx], &s[n*bx]);
    return;
}

void test_gpstep(int n, int gridSize)
{
    int totalMemSize;
    double *x, *xl, *xu, *alpha, *w, *s;

    printf("%-25s", "Testing gpstep()  . . .");
    fflush(stdout);

    std::mt19937 gen(0);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    totalMemSize = sizeof(double)*gridSize*n;
    x = (double *)malloc(totalMemSize);
    xl = (double *)malloc(totalMemSize);
    xu = (double *)malloc(totalMemSize);
    alpha = (double *)malloc(sizeof(double)*gridSize);
    w = (double *)malloc(totalMemSize);
    s = (double *)malloc(totalMemSize);

    for (int i = 0; i < gridSize; i++) {
        alpha[i] = dist(gen);
        for (int j = 0; j < n ; j++) {
            x[n*i + j] = dist(gen);
            xl[n*i + j] = x[n*i + j] - dist(gen);
            xu[n*i + j] = x[n*i + j] + dist(gen);
            w[n*i + j] = dist(gen);

            int k = rand() % 2;
            if (k == 0) {
                if (x[n*i + j] + alpha[i]*w[n*i + j] > xl[n*i + j]) {
                    w[n*i + j] = (xl[n*i + j] - x[n*i + j]) / alpha[i] - 0.1;
                }
            } else {
                if (x[n*i + j] + alpha[i]*w[n*i + j] < xl[n*i + j]) {
                    w[n*i + j] = (xu[n*i + j] - x[n*i + j]) / alpha[i] + 0.1;
                }
            }
        }
        gpstep(n, &x[n*i], &xl[n*i], &xu[n*i], alpha[i], &w[n*i], &s[n*i]);
    }

    double *devX, *devXl, *devXu, *devAlpha, *devW, *devS;
    cudaMalloc(&devX, totalMemSize);
    cudaMalloc(&devXl, totalMemSize);
    cudaMalloc(&devXu, totalMemSize);
    cudaMalloc(&devAlpha, sizeof(double)*gridSize);
    cudaMalloc(&devW, totalMemSize);
    cudaMalloc(&devS, totalMemSize);
    cudaMemcpy(devX, x, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devXl, xl, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devXu, xu, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devAlpha, alpha, sizeof(double)*gridSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devW, w, totalMemSize, cudaMemcpyHostToDevice);

    gpstep_kernel<<<gridSize, dim3(n,n)>>>(n, devX, devXl, devXu, devAlpha, devW, devS);
    cudaDeviceSynchronize();

    double *hostS = (double *)malloc(totalMemSize);
    cudaMemcpy(hostS, devS, totalMemSize, cudaMemcpyDeviceToHost);

    double err = 0;
    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < n; j++) {
            err = max(err, abs(hostS[n*i + j] - s[n*i + j]));
        }
    }
    printf("%.5e\n", err);

    free(x);
    free(xl);
    free(xu);
    free(alpha);
    free(w);
    free(s);
    free(hostS);
    cudaFree(devX);
    cudaFree(devXl);
    cudaFree(devXu);
    cudaFree(devAlpha);
    cudaFree(devW);
    cudaFree(devS);

    return;
}