__global__
void ssyax_kernel(int n, double *A, double *z, double *q)
{
    int bx = blockIdx.x;

    cssyax(n, &A[(n*n)*bx], &z[n*bx], &q[n*bx]);
    return;
}

void test_ssyax(int n, int gridSize)
{
    double *A, *z, *q;

    printf("%-25s", "Testing ssyax()   . . .");
    fflush(stdout);

    std::mt19937 gen(0);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    A = (double *)malloc(sizeof(double)*gridSize*(n*n));
    z = (double *)malloc(sizeof(double)*gridSize*n);
    q = (double *)malloc(sizeof(double)*gridSize*n);

    int matBlockSize = n*n;
    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < n; j++) {
            z[n*i + j] = dist(gen);
            for (int k = 0; k < n; k++) {
                A[matBlockSize*i + n*j + k] = dist(gen);
            }
        }
        gemv(n, &q[n*i], 1.0, &A[matBlockSize*i], &z[n*i], 0.0, NULL);
    }

    double *devA, *devZ, *devQ;
    cudaMalloc(&devA, sizeof(double)*gridSize*(n*n));
    cudaMalloc(&devZ, sizeof(double)*gridSize*n);
    cudaMalloc(&devQ, sizeof(double)*gridSize*n);
    cudaMemcpy(devA, A, sizeof(double)*gridSize*(n*n), cudaMemcpyHostToDevice);
    cudaMemcpy(devZ, z, sizeof(double)*gridSize*n, cudaMemcpyHostToDevice);

    ssyax_kernel<<<gridSize, dim3(n,n)>>>(n, devA, devZ, devQ);
    cudaDeviceSynchronize();

    double *hostQ = (double *)malloc(sizeof(double)*gridSize*n);
    cudaMemcpy(hostQ, devQ, sizeof(double)*gridSize*n, cudaMemcpyDeviceToHost);

    double err = 0;
    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < n; j++) {
            err = max(err, abs(hostQ[n*i + j] - q[n*i + j]));
        }
    }
    printf("%.5e\n", err);

    free(A);
    free(z);
    free(q);
    cudaFree(devA);
    cudaFree(devZ);
    cudaFree(devQ);

    return;
}