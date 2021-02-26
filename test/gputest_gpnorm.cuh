__global__
void gpnorm_kernel(int n, double *x, double *xl, double *xu, double *g, double *v)
{
    int bx = blockIdx.x;

    v[bx] = cgpnorm(n, &x[n*bx], &xl[n*bx], &xu[n*bx], &g[n*bx]);
    return;
}

void test_gpnorm(int n, int gridSize)
{
    int totalMemSize;
    double *x, *xl, *xu, *g, *v;

    printf("%-25s", "Testing gpnorm()  . . .");
    fflush(stdout);

    std::mt19937 gen(0);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    totalMemSize = sizeof(double)*gridSize*n;
    x = (double *)malloc(totalMemSize);
    xl = (double *)malloc(totalMemSize);
    xu = (double *)malloc(totalMemSize);
    g = (double *)malloc(totalMemSize);
    v = (double *)malloc(sizeof(double)*gridSize);

    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < n; j++) {
            x[n*i + j] = dist(gen);
            xl[n*i + j] = x[n*i + j] - dist(gen);
            xu[n*i + j] = x[n*i + j] + dist(gen);
            g[n*i + j] = 2*dist(gen) - 1;
        }
    }

    double *devX, *devXl, *devXu, *devG, *devV;
    cudaMalloc(&devX, totalMemSize);
    cudaMalloc(&devXl, totalMemSize);
    cudaMalloc(&devXu, totalMemSize);
    cudaMalloc(&devG, totalMemSize);
    cudaMalloc(&devV, sizeof(double)*gridSize);
    cudaMemcpy(devX, x, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devXl, xl, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devXu, xu, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devG, g, totalMemSize, cudaMemcpyHostToDevice);

    gpnorm_kernel<<<gridSize, dim3(n,n)>>>(n, devX, devXl, devXu, devG, devV);

    double *hostV = (double *)malloc(sizeof(double)*gridSize);
    cudaMemcpy(hostV, devV, sizeof(double)*gridSize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < gridSize; i++) {
        v[i] = gpnorm(n, &x[n*i], &xl[n*i], &xu[n*i], &g[n*i]);
    }

    double err = 0;
    for (int i = 0; i < gridSize; i++) {
        err = max(err, abs(hostV[i] - v[i]));
    }
    printf("%.5e\n", err);

    free(x);
    free(xl);
    free(xu);
    free(g);
    free(hostV);
    cudaFree(devX);
    cudaFree(devXl);
    cudaFree(devXu);
    cudaFree(devG);

    return;
}