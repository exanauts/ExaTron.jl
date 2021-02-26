__global__
void trqsol_kernel(int n, double *x, double *p, double *delta, double *sigma)
{
    int bx = blockIdx.x;

    sigma[bx] = ctrqsol(n, &x[n*bx], &p[n*bx], delta[bx]);
    return;
}

void test_trqsol(int n, int gridSize)
{
    int totalMemSize;
    double *x, *p, *sigma, *delta;

    printf("%-25s", "Testing trqsol()  . . .");
    fflush(stdout);

    std::mt19937 gen(0);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    totalMemSize = sizeof(double)*gridSize*n;
    x = (double *)malloc(totalMemSize);
    p = (double *)malloc(totalMemSize);
    sigma = (double *)malloc(sizeof(double)*gridSize);
    delta = (double *)malloc(sizeof(double)*gridSize);

    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < n; j++) {
            x[n*i + j] = dist(gen);
            p[n*i + j] = dist(gen);
        }
        sigma[i] = dist(gen);
        delta[i] = 0;
        for (int j = 0; j < n; j++) {
            delta[i] += (x[n*i + j] + sigma[i]*p[n*i + j])*(x[n*i + j] + sigma[i]*p[n*i + j]);
        }
        delta[i] = sqrt(delta[i]);
    }

    double *devX, *devP, *devSigma, *devDelta;

    cudaMalloc(&devX, totalMemSize);
    cudaMalloc(&devP, totalMemSize);
    cudaMalloc(&devSigma, sizeof(double)*gridSize);
    cudaMalloc(&devDelta, sizeof(double)*gridSize);
    cudaMemcpy(devX, x, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devP, p, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devDelta, delta, sizeof(double)*gridSize, cudaMemcpyHostToDevice);

    trqsol_kernel<<<gridSize, dim3(n,n)>>>(n, devX, devP, devDelta, devSigma);
    cudaDeviceSynchronize();

    double *hostSigma = (double *)malloc(sizeof(double)*gridSize);
    cudaMemcpy(hostSigma, devSigma, sizeof(double)*gridSize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < gridSize; i++) {
        sigma[i] = trqsol(n, &x[n*i], &p[n*i], delta[i]);
    }

    double err = 0;
    for (int i = 0; i < gridSize; i++) {
        err = max(err, abs(hostSigma[i] - sigma[i]));
    }
    printf("%.5e\n", err);

    free(x);
    free(p);
    free(delta);
    free(sigma);
    free(hostSigma);
    cudaFree(devX);
    cudaFree(devP);
    cudaFree(devSigma);
    cudaFree(devDelta);

    return;
}