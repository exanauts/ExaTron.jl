__global__
void nrm2_kernel(int n, double *a, double *v)
{
    int bx = blockIdx.x;

    v[bx] = cnrm2(n, &a[n*bx]);
    return;
}

void test_nrm2(int n, int gridSize)
{
    int totalMemSize;
    double *a, *v;

    printf("%-25s", "Testing nrm2()    . . .");
    fflush(stdout);

    std::mt19937 gen(0);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    totalMemSize = sizeof(double)*gridSize*n;
    a = (double *)malloc(totalMemSize);
    v = (double *)malloc(sizeof(double)*gridSize);

    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < n; j++) {
            a[n*i + j] = dist(gen);
        }
        v[i] = 0;
        for (int j = 0; j < n; j++) {
            v[i] += a[n*i + j]*a[n*i + j];
        }
        v[i] = sqrt(v[i]);
    }

    double *devA, *devV;
    cudaMalloc(&devA, totalMemSize);
    cudaMalloc(&devV, sizeof(double)*gridSize);
    cudaMemcpy(devA, a, totalMemSize, cudaMemcpyHostToDevice);

    nrm2_kernel<<<gridSize, dim3(n,n)>>>(n, devA, devV);
    cudaDeviceSynchronize();

    double *hostV = (double *)malloc(sizeof(double)*gridSize);
    cudaMemcpy(hostV, devV, sizeof(double)*gridSize, cudaMemcpyDeviceToHost);

    double err = 0;
    for (int i = 0; i < gridSize; i++) {
        err = max(err, abs(hostV[i] - v[i]));
    }
    printf("%.5e\n", err);

    free(a);
    free(v);
    cudaFree(devA);
    cudaFree(devV);

    return;
}