#pragma once

#include <cuda.h>
#include <cub/util_allocator.cuh>

using namespace std;
using namespace cub;

#include "sharedmem.cuh"

#define NUM_ELEM    8

template <typename T>
__global__ void precond_cg(unsigned int n, T* __restrict__ in)
{
    const int t = threadIdx.x;
    const int wg = blockDim.x;
    const int gid = blockIdx.x;

    T x[NUM_ELEM];
    return;
}

template <typename T>
void runExaTron(unsigned int n, T *d_in, T *d_out,
    CachingDeviceAllocator &g_allocator)
{
    return;
}
