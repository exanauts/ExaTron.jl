__global__
void copy_kernel(int n, double *src, double *dest)
{
    int bx = blockIdx.x;

    ccopy(n, &src[n*bx], &dest[n*bx]);
    return;
}

void test_copy(int n, int gridSize)
{
    int totalMemSize;
    double *src, *dest, *devSrc, *devDest;

    printf("%-25s", "Testing copy()    . . .");
    fflush(stdout);

    std::mt19937 gen(0);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    totalMemSize = sizeof(double)*gridSize*n;
    src = (double *)malloc(totalMemSize);
    dest = (double *)malloc(totalMemSize);
    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < n; j++) {
            src[n*i + j] = dist(gen);
        }
    }
    memcpy(dest, src, totalMemSize);

    cudaMalloc(&devSrc, totalMemSize);
    cudaMalloc(&devDest, totalMemSize);
    cudaMemcpy(devSrc, src, totalMemSize, cudaMemcpyHostToDevice);

    copy_kernel<<<gridSize, dim3(n,n)>>>(n, devSrc, devDest);
    cudaDeviceSynchronize();

    double *hostDest = (double *)malloc(totalMemSize);
    cudaMemcpy(hostDest, devDest, totalMemSize, cudaMemcpyDeviceToHost);

    double err = 0;
    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < n; j++) {
            err = max(err, abs(hostDest[n*i + j] - dest[n*i + j]));
        }
    }
    printf("%.5e\n", err);

    free(src);
    free(dest);
    free(hostDest);
    cudaFree(devSrc);
    cudaFree(devDest);

    return;
}