__global__
void dot_kernel(int n, double *a, double *b, double *v)
{
    int bx = blockIdx.x;

    v[bx] = cdot(n, &a[n*bx], &b[n*bx]);
    return;
}

void test_dot(int n, int gridSize)
{
    int totalMemSize;
    double *a, *b, *v;

    printf("%-25s", "Testing dot()     . . .");
    fflush(stdout);

    std::mt19937 gen(0);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    totalMemSize = sizeof(double)*gridSize*n;
    a = (double *)malloc(totalMemSize);
    b = (double *)malloc(totalMemSize);
    v = (double *)malloc(sizeof(double)*gridSize);

    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < n; j++) {
            a[n*i + j] = dist(gen);
            b[n*i + j] = dist(gen);
        }
        v[i] = 0;
        for (int j = 0; j < n; j++) {
            v[i] += a[n*i + j]*b[n*i + j];
        }
    }

    double *devA, *devB, *devV;
    cudaMalloc(&devA, totalMemSize);
    cudaMalloc(&devB, totalMemSize);
    cudaMalloc(&devV, sizeof(double)*gridSize);
    cudaMemcpy(devA, a, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, b, totalMemSize, cudaMemcpyHostToDevice);

    dot_kernel<<<gridSize, dim3(n,n)>>>(n, devA, devB, devV);
    cudaDeviceSynchronize();

    double *hostV = (double *)malloc(sizeof(double)*gridSize);
    cudaMemcpy(hostV, devV, sizeof(double)*gridSize, cudaMemcpyDeviceToHost);

    double err = 0;
    for (int i = 0; i < gridSize; i++) {
        err = max(err, abs(hostV[i] - v[i]));
    }
    printf("%.5e\n", err);

    free(a);
    free(b);
    free(v);
    free(hostV);
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devV);

    return;
}