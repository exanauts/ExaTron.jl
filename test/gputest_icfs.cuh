__global__
void cicfs_kernel(int n, double alpha, double *dev, double *result)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;

    extern __shared__ double s[];

    double *A = s;
    double *L = &A[n*n];
    double *wa1 = &L[n*n];
    double *wa2 = &wa1[n];

    int start = bx*(n*n);
    A[n*ty + tx] = dev[start + n*ty + tx];
    __syncthreads();

    cicfs(n, alpha, A, L, wa1, wa2);

    result[start + n*ty + tx] = L[n*ty + tx];
    __syncthreads();

    return;
}

void test_icfs(int n, int gridSize)
{
    int threadSize, totalMemSize;
    double *L, *A, *dev, *dev_result, *host_result;

    printf("%-25s", "Testing icfs()    . . .");
    fflush(stdout);

    std::mt19937 gen(0);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    threadSize = n*n;
    totalMemSize = gridSize*threadSize*sizeof(double);

    L = (double *)malloc(totalMemSize);
    memset(L, 0, totalMemSize);
    for (int i = 0; i < gridSize; i++) {
        double *Li = &L[threadSize*i];
        for (int j = 0; j < n; j++) {
            for (int k = j; k < n; k++) {
                Li[j*n + k] = 1e-1 + dist(gen);
            }
        }
    }

    A = (double *)malloc(totalMemSize);
    memset(A, 0, totalMemSize);
    for (int i = 0; i < gridSize; i++) {
        llt(n, &A[threadSize*i], &L[threadSize*i]);
    }

    host_result = (double *)malloc(totalMemSize);

    cudaMalloc(&dev, totalMemSize);
    cudaMalloc(&dev_result, totalMemSize);
    cudaMemcpy(dev, A, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(host_result, dev, totalMemSize, cudaMemcpyDeviceToHost);

    double alpha = 1.0;
    cicfs_kernel<<<gridSize, dim3(n,n), (2*n + 2*n*n)*sizeof(double)>>>(n, alpha, dev, dev_result);
    cudaDeviceSynchronize();
    cudaMemcpy(host_result, dev_result, totalMemSize, cudaMemcpyDeviceToHost);

    double *B, *wa1, *wa2;
    B = (double *)malloc(totalMemSize);
    wa1 = (double *)malloc(sizeof(double)*n);
    wa2 = (double *)malloc(sizeof(double)*n);

    for (int i = 0; i < gridSize; i++) {
        icfs(n, alpha, &A[threadSize*i], &B[threadSize*i], wa1, wa2);
    }

    double derr = 0;
    for (int i = 0; i < gridSize; i++) {
        double *Hi = &B[threadSize*i];
        double *Di = &host_result[threadSize*i];
        for (int j = 0; j < n; j++) {
            for (int k = j; k < n; k++) {
                derr = max(derr, fabs(Hi[n*j + k] - Di[n*j + k]));
            }
        }
    }
    printf("%.5e\n", derr);

    cudaFree(dev);
    cudaFree(dev_result);
    free(L);
    free(A);
    free(B);
    free(wa1);
    free(wa2);

    return;
}