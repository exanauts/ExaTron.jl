__global__
void scal_kernel(int n, double *s, double *a)
{
    int bx = blockIdx.x;

    cscal(n, s[bx], &a[n*bx]);
    return;
}

void test_scal(int n, int gridSize)
{
    int totalMemSize;
    double *a, *s;

    printf("%-25s", "Testing scal()    . . .");

    std::mt19937 gen(0);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    totalMemSize = sizeof(double)*gridSize*n;
    a = (double *)malloc(totalMemSize);
    s = (double *)malloc(sizeof(double)*gridSize);

    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < n; j++) {
            a[n*i + j] = dist(gen);
        }
        s[i] = dist(gen);
    }

    double *devA, *devS;
    cudaMalloc(&devA, totalMemSize);
    cudaMalloc(&devS, sizeof(double)*gridSize);
    cudaMemcpy(devA, a, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devS, s, sizeof(double)*gridSize, cudaMemcpyHostToDevice);

    scal_kernel<<<gridSize, dim3(n,n)>>>(n, devS, devA);
    cudaDeviceSynchronize();

    double *hostA;
    hostA = (double *)malloc(totalMemSize);
    cudaMemcpy(hostA, devA, totalMemSize, cudaMemcpyDeviceToHost);

    double err = 0;
    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < n; j++) {
            err = max(err, abs(hostA[n*i + j] - s[i]*a[n*i + j]));
        }
    }
    printf("%.5e\n", err);

    free(a);
    free(s);
    free(hostA);
    cudaFree(devA);
    cudaFree(devS);

    return;
}