__global__
void axpy_kernel(int n, double *a, double *x, double *y)
{
    int bx = blockIdx.x;

    caxpy(n, a[bx], &x[n*bx], &y[n*bx]);
    return;
}

void test_axpy(int n, int gridSize)
{
    int totalMemSize;
    double *a, *x, *y;

    printf("%-25s", "Testing axpy()    . . .");
    fflush(stdout);

    std::mt19937 gen(0);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    totalMemSize = sizeof(double)*gridSize*n;
    a = (double *)malloc(sizeof(double)*gridSize);
    x = (double *)malloc(totalMemSize);
    y = (double *)malloc(totalMemSize);

    for (int i = 0; i < gridSize; i++) {
        a[i] = dist(gen);
        for (int j = 0; j < n; j++) {
            x[n*i + j] = dist(gen);
            y[n*i + j] = dist(gen);
        }
    }

    double *devA, *devX, *devY;
    cudaMalloc(&devA, sizeof(double)*gridSize);
    cudaMalloc(&devX, totalMemSize);
    cudaMalloc(&devY, totalMemSize);
    cudaMemcpy(devA, a, sizeof(double)*gridSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devX, x, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devY, y, totalMemSize, cudaMemcpyHostToDevice);

    axpy_kernel<<<gridSize, dim3(n,n)>>>(n, devA, devX, devY);
    cudaDeviceSynchronize();

    double *hostY = (double *)malloc(totalMemSize);
    cudaMemcpy(hostY, devY, totalMemSize, cudaMemcpyDeviceToHost);

    double err = 0;
    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < n; j++) {
            err = max(err, abs(hostY[n*i + j] - (y[n*i + j] + a[i]*x[n*i+ j])));
        }
    }
    printf("%.5e\n", err);

    free(a);
    free(x);
    free(y);
    free(hostY);
    cudaFree(devA);
    cudaFree(devX);
    cudaFree(devY);

    return;
}