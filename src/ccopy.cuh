__device__
void ccopy(int n, double *src, double *dest)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (tx < n && ty == 0) {
        dest[tx] = src[tx];
    }
    __syncthreads();
    return;
}