__global__
void breakpt_kernel(int n, double *x, double *xl, double *xu, double *w,
                    int *nbrpt, double *brptmin, double *brptmax)
{
    int bx = blockIdx.x;

    cbreakpt(n, &x[n*bx], &xl[n*bx], &xu[n*bx], &w[n*bx],
             &nbrpt[bx], &brptmin[bx], &brptmax[bx]);
    return;
}

void test_breakpt(int n, int gridSize)
{
    int totalMemSize;
    int *nbrpt;
    double *x, *xl, *xu, *w, *brptmin, *brptmax;

    printf("%-25s", "Testing breakpt() . . .");
    fflush(stdout);

    std::mt19937 gen(0);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    totalMemSize = sizeof(double)*gridSize*n;
    x = (double *)malloc(totalMemSize);
    xl = (double *)malloc(totalMemSize);
    xu = (double *)malloc(totalMemSize);
    w = (double *)malloc(totalMemSize);
    nbrpt = (int *)malloc(sizeof(int)*gridSize);
    brptmin = (double *)malloc(sizeof(double)*gridSize);
    brptmax = (double *)malloc(sizeof(double)*gridSize);

    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < n; j++) {
            x[n*i + j] = dist(gen);
            xl[n*i + j] = x[n*i + j] - dist(gen);
            xu[n*i + j] = x[n*i + j] + dist(gen);
            w[n*i + j] = 2*dist(gen) - 1.0;
        }
    }

    for (int i = 0; i < gridSize; i++) {
        breakpt(n, &x[n*i], &xl[n*i], &xu[n*i], &w[n*i], &nbrpt[i], &brptmin[i], &brptmax[i]);
    }

    int *devNbrpt;
    double *devX, *devXl, *devXu, *devW, *devBrptmin, *devBrptmax;

    cudaMalloc(&devX, totalMemSize);
    cudaMalloc(&devXl, totalMemSize);
    cudaMalloc(&devXu, totalMemSize);
    cudaMalloc(&devW, totalMemSize);
    cudaMalloc(&devNbrpt, sizeof(int)*gridSize);
    cudaMalloc(&devBrptmin, sizeof(double)*gridSize);
    cudaMalloc(&devBrptmax, sizeof(double)*gridSize);
    cudaMemcpy(devX, x, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devXl, xl, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devXu, xu, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devW, w, totalMemSize, cudaMemcpyHostToDevice);

    breakpt_kernel<<<gridSize, dim3(n,n)>>>(n, devX, devXl, devXu, devW, devNbrpt, devBrptmin, devBrptmax);
    cudaDeviceSynchronize();

    int *hostNbrpt;
    double *hostBrptmin, *hostBrptmax;

    hostNbrpt = (int *)malloc(sizeof(int)*gridSize);
    hostBrptmin = (double *)malloc(sizeof(double)*gridSize);
    hostBrptmax = (double *)malloc(sizeof(double)*gridSize);
    cudaMemcpy(hostNbrpt, devNbrpt, sizeof(int)*gridSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostBrptmin, devBrptmin, sizeof(double)*gridSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostBrptmax, devBrptmax, sizeof(double)*gridSize, cudaMemcpyDeviceToHost);

    int ierr = 0;
    double err = 0;
    for (int i = 0; i < gridSize; i++) {
        ierr = max(ierr, abs(hostNbrpt[i] - nbrpt[i]));
        err = max(err, abs(hostBrptmin[i] - brptmin[i]));
        err = max(err, abs(hostBrptmax[i] - brptmax[i]));
    }
    printf("%.5e\n", max(err, (double)ierr));

    free(x);
    free(xl);
    free(xu);
    free(w);
    free(nbrpt);
    free(brptmin);
    free(brptmax);
    cudaFree(devNbrpt);
    cudaFree(devBrptmin);
    cudaFree(devBrptmax);
    cudaFree(devX);
    cudaFree(devXl);
    cudaFree(devXu);
    cudaFree(devW);

    return;
}