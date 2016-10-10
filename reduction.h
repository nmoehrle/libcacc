/*
 * Copyright (C) 2015, Nils Moehrle
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef CACC_REDUCTION_HEADER
#define CACC_REDUCTION_HEADER

#include "defines.h"

#include "array.h"
#include "variable.h"

CACC_NAMESPACE_BEGIN

#ifndef REDUCTION_BLOCK_SIZE
    #define REDUCTION_BLOCK_SIZE 256
#endif

__inline__ __device__
float sum_warp(float val) {
    #pragma unroll
    for (int o = warpSize / 2; o > 0; o /= 2) {
        val += __shfl_down(val, o);
    }
    return val;
}

__global__
void sum_kernel(float * values, float * ret, uint n)
{
    __shared__ float sm[32];
    uint lane = threadIdx.x % warpSize;
    uint tx = threadIdx.x;
    uint wx = threadIdx.x / warpSize;
    uint gstride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (uint i = blockIdx.x * blockDim.x + tx; i < n; i += gstride) {
        sum += values[i];
    }

    sum = sum_warp(sum);

    if (lane == 0) sm[wx] = sum;

    __syncthreads();

    sum = (threadIdx.x < blockDim.x / warpSize) ? sm[lane] : 0.0f;

    if (wx == 0) sum = sum_warp(sum);

    if (tx == 0) atomicAdd(ret, sum);
}

float sum(Array<float, DEVICE>::Ptr darray) {
    Array<float, DEVICE>::Data data = darray->cdata();

    uint num_threads = cacc::divup(data.num_values, std::log2(data.num_values));
    uint num_blocks = cacc::divup(num_threads, REDUCTION_BLOCK_SIZE);

    Variable<float, DEVICE> dret(0.0f);

    dim3 grid(num_blocks);
    dim3 block(REDUCTION_BLOCK_SIZE);
    sum_kernel<<<grid, block>>>(
        data.data_ptr, dret.cptr(), data.num_values
    );

    Variable<float, HOST> ret(dret);

    return ret.ref();
}

CACC_NAMESPACE_END

#endif /* CACC_REDUCTION_HEADER */
