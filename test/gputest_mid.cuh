__global__
void mid_kernel(int n, double *x, double *xl, double *xu)
{
    int bx = blockIdx.x;

    cmid(n, &x[n*bx], &xl[n*bx], &xu[n*bx]);
    return;
}

void test_mid(int n, int gridSize)
{
    int totalMemSize;
    double *x, *xl, *xu;

    printf("%-25s", "Testing mid()     . . .");
    fflush(stdout);

    std::mt19937 gen(0);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    totalMemSize = sizeof(double)*gridSize*n;
    x = (double *)malloc(totalMemSize);
    xl = (double *)malloc(totalMemSize);
    xu = (double *)malloc(totalMemSize);

    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < n; j++) {
            x[n*i + j] = dist(gen);
            xl[n*i + j] = x[n*i + j] - dist(gen);
            xu[n*i + j] = x[n*i + j] + dist(gen);
        }
    }

    double *devX, *devXl, *devXu;
    cudaMalloc(&devX, totalMemSize);
    cudaMalloc(&devXl, totalMemSize);
    cudaMalloc(&devXu, totalMemSize);
    cudaMemcpy(devX, x, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devXl, xl, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devXu, xu, totalMemSize, cudaMemcpyHostToDevice);

    mid_kernel<<<gridSize, dim3(n,n)>>>(n, devX, devXl, devXu);
    cudaDeviceSynchronize();

    double *hostX = (double *)malloc(totalMemSize);
    cudaMemcpy(hostX, devX, totalMemSize, cudaMemcpyDeviceToHost);

    double err = 0;
    for (int i = 0; i < gridSize; i++) {
        mid(n, &x[n*i], &xl[n*i], &xu[n*i]);
        err = max(err, diff_vector(n, &hostX[n*i], &x[n*i]));
    }
    printf("%.5e\n", err);

    free(x);
    free(xl);
    free(xu);
    free(hostX);
    cudaFree(devX);
    cudaFree(devXl);
    cudaFree(devXu);

    return;
}