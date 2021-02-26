__global__
void cicf_kernel(int n, double *dev, double *result)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;

    extern __shared__ double s[];

    double *A = s;

    int start = bx*(n*n);
    A[n*ty + tx] = dev[start + n*ty + tx];
    __syncthreads();

    cicf(n, A);

    result[start + n*ty + tx] = A[n*ty + tx];
    __syncthreads();

    return;
}

void test_icf(int n, int gridSize)
{
    int threadSize, totalMemSize;
    double *L, *A, *dev, *dev_result, *host_result;

    printf("%-25s", "Testing icf()     . . .");
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

    cicf_kernel<<<gridSize, dim3(n,n), (n*n)*sizeof(double)>>>(n, dev, dev_result);
    cudaDeviceSynchronize();
    cudaMemcpy(host_result, dev_result, totalMemSize, cudaMemcpyDeviceToHost);

    memcpy(L, A, totalMemSize);
    for (int i = 0; i < gridSize; i++) {
        icf_right(n, &L[threadSize*i]);
    }

    double derr = 0;
    for (int i = 0; i < gridSize; i++) {
        double *Ai = &A[threadSize*i];
        double *Di = &host_result[threadSize*i];
        for (int j = 0; j < n; j++) {
            for (int k = j; k < n; k++) {
                double dval = 0;
                for (int p = 0; p <= j; p++) {
                    dval += Di[n*p + k]*Di[n*p + j];
                }
                derr = max(derr, fabs(Ai[n*j + k] - dval));
            }
        }
    }
    printf("%.5e\n", derr);

    cudaFree(dev);
    cudaFree(dev_result);
    free(L);
    free(A);

    return;
}