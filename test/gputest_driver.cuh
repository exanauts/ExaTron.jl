static double *_hostA = NULL;
static double *_hostC = NULL;

double host_eval_f(int n, double *x, int bx)
{
    double *A = &_hostA[(n*n)*bx];
    double *c = &_hostC[n*bx];
    double f = 0;

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            f += x[i]*A[n*j + i]*x[j];
        }
    }
    f *= 0.5;
    for (int j = 0; j < n; j++) {
        f += x[j]*c[j];
    }

    return f;
}

void host_eval_g(int n, double *x, double *g, int bx)
{
    double *A = &_hostA[(n*n)*bx];
    double *c = &_hostC[n*bx];

    memset(g, 0, sizeof(double)*n);
    for (int i = 0; i < n; i++) {
        double gval = 0;
        for (int j = 0; j < n; j++) {
            gval += A[n*j + i]*x[j];
        }
        g[i] = gval + c[i];
    }

    return;
}

void host_eval_h(int n, double *x, double *H, int bx)
{
    double *A = &_hostA[(n*n)*bx];

    memcpy(H, A, sizeof(double)*(n*n));
    return;
}

__device__ double *_devA = NULL;
__device__ double *_devC = NULL;

__device__
double dev_eval_f(int n, double *x)
{
    int bx = blockIdx.x;
    double *A = &_devA[(n*n)*bx];
    double *c = &_devC[n*bx];
    double f = 0;

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            f += x[i]*A[n*j + i]*x[j];
        }
    }
    f *= 0.5;
    for (int j = 0; j < n; j++) {
        f += x[j]*c[j];
    }

    __syncthreads();
    return f;
}

__device__
void dev_eval_g(int n, double *x, double *g)
{
    int bx = blockIdx.x;
    double *A = &_devA[(n*n)*bx];
    double *c = &_devC[n*bx];

    memset(g, 0, sizeof(double)*n);
    for (int i = 0; i < n; i++) {
        double gval = 0;
        for (int j = 0; j < n; j++) {
            gval += A[n*j + i]*x[j];
        }
        g[i] = gval + c[i];
    }

    __syncthreads();
    return;
}

__device__
void dev_eval_h(int n, double *x, double *H)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    double *A = &_devA[(n*n)*bx];

    H[n*ty + tx] = A[n*ty + tx];
    __syncthreads();
    return;
}

__global__
void driver_kernel(int n, int max_feval, int max_minor,
                   int *_status, int *_minor_iter,
                   double *_x, double *_xl, double *_xu)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int status, minor_iter;
    double *x, *xl, *xu;

    extern __shared__ double shmem[];
    x = shmem;
    xl = x + n;
    xu = xl + n;
    if (ty == 0) {
        x[tx] = _x[n*bx + tx];
        xl[tx] = _xl[n*bx + tx];
        xu[tx] = _xu[n*bx + tx];
    }
    __syncthreads();

    cdriver(n, max_feval, max_minor, &status, &minor_iter,
            x, xl, xu, &dev_eval_f, &dev_eval_g, &dev_eval_h);

    if (ty == 0) {
        _x[n*bx + tx] = x[tx];
    }
    _status[bx] = status;
    _minor_iter[bx] = minor_iter;

    return;
}

void test_driver(int n, int gridSize)
{
    int blockSize, totalMemSize;
    int *status, *minor_iter;
    double *x, *xl, *xu, *c, *L, *A;

    printf("%-25s", "Testing driver()  . . .");
    fflush(stdout);

    std::mt19937 gen(0);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    blockSize = n*n;
    totalMemSize = sizeof(double)*gridSize*n;
    x = (double *)malloc(totalMemSize);
    xl = (double *)malloc(totalMemSize);
    xu = (double *)malloc(totalMemSize);
    c = (double *)malloc(totalMemSize);
    L = (double *)malloc(sizeof(double)*gridSize*blockSize);
    A = (double *)malloc(sizeof(double)*gridSize*blockSize);
    status = (int *)malloc(sizeof(int)*gridSize);
    minor_iter = (int *)malloc(sizeof(int)*gridSize);

    memset(L, 0, sizeof(double)*gridSize*blockSize);
    for (int i = 0; i < gridSize; i++) {
        double *Li = &L[blockSize*i];
        for (int j = 0; j < n; j++) {
            for (int k = j; k < n; k++) {
                Li[j*n + k] = 1e-1 + dist(gen);
            }
        }
    }

    memset(A, 0, sizeof(double)*gridSize*blockSize);
    for (int i = 0; i < gridSize; i++) {
        llt(n, &A[blockSize*i], &L[blockSize*i]);
    }

    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < n; j++) {
            x[n*i + j] = dist(gen);
            xl[n*i + j] = x[n*i + j] - dist(gen);
            xu[n*i + j] = x[n*i + j] + dist(gen);
            c[n*i + j] = dist(gen);
        }
    }

    double *devX, *devXl, *devXu, *devC, *devA;
    int *devStatus, *devMinorIter;

    cudaMalloc(&devX, totalMemSize);
    cudaMalloc(&devXl, totalMemSize);
    cudaMalloc(&devXu, totalMemSize);
    cudaMalloc(&devC, totalMemSize);
    cudaMalloc(&devA, sizeof(double)*gridSize*blockSize);
    cudaMalloc(&devStatus, sizeof(int)*gridSize);
    cudaMalloc(&devMinorIter, sizeof(int)*gridSize);
    cudaMemcpy(devX, x, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devXl, xl, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devXu, xu, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devA, A, sizeof(double)*gridSize*blockSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devC, c, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(_devA, &devA, sizeof(double *), size_t(0), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(_devC, &devC, sizeof(double *), size_t(0), cudaMemcpyHostToDevice);

    int max_feval = 500, max_minor = 200;
    int shmem_size = sizeof(int)*(4*n) + sizeof(double)*(14*n + 3*blockSize);

    driver_kernel<<<gridSize, dim3(n,n), shmem_size>>>(n, max_feval, max_minor, devStatus,
                                                       devMinorIter, devX, devXl, devXu);
    cudaDeviceSynchronize();

    int *hostStatus = (int *)malloc(sizeof(int)*gridSize);
    int *hostMinorIter = (int *)malloc(sizeof(int)*gridSize);
    double *hostX = (double *)malloc(totalMemSize);
    cudaMemcpy(hostStatus, devStatus, sizeof(int)*gridSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostMinorIter, devMinorIter, sizeof(int)*gridSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostX, devX, totalMemSize, cudaMemcpyDeviceToHost);

    _hostA = A;
    _hostC = c;
    for (int i = 0; i < gridSize; i++) {
        driver(n, max_feval, max_minor, &status[i], &minor_iter[i],
               &x[n*i], &xl[n*i], &xu[n*i],
               &host_eval_f, &host_eval_g, &host_eval_h, i);
    }

    double err = 0;
    for (int i = 0; i < gridSize; i++) {
        err = max(err, diff_vector(n, &hostX[n*i], &x[n*i]));
    }
    printf("%.5e\n", err);

    free(x);
    free(xl);
    free(xu);
    free(c);
    free(status);
    free(minor_iter);
    free(L);
    free(A);
    free(hostX);
    cudaFree(devX);
    cudaFree(devXl);
    cudaFree(devXu);
    cudaFree(devC);
    cudaFree(devA);
    cudaFree(devStatus);
    cudaFree(devMinorIter);

    return;
}