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

CACC_NAMESPACE_BEGIN

#ifndef REDUCTION_BLOCK_SIZE
    #define REDUCTION_BLOCK_SIZE 256
#endif

/* Derived from the slides "Optimizing Parallel Reduction in CUDA"
 * by Mark Harris while at NVIDIA in 2007 */
template <uint block_size>
__global__ void sum(float * values, float * ret, uint n)
{
    __shared__ volatile float sm[block_size];
    uint tx = threadIdx.x;
    uint i = blockIdx.x * blockDim.x + tx;
    uint gstride = blockDim.x * gridDim.x;
    sm[tx] = 0.0f;

    while (i < n) {
        sm[tx] += values[i];
        i += gstride;
    }
    __syncthreads();

    if (block_size >= 512) { if (tx < 256) { sm[tx] += sm[tx + 256]; } __syncthreads(); }
    if (block_size >= 256) { if (tx < 128) { sm[tx] += sm[tx + 128]; } __syncthreads(); }
    if (block_size >= 128) { if (tx < 64) { sm[tx] += sm[tx + 64]; } __syncthreads(); }
    if (tx < 32) {
        if (block_size >= 64) sm[tx] += sm[tx + 32];
        if (block_size >= 32) sm[tx] += sm[tx + 16];
        if (block_size >= 16) sm[tx] += sm[tx + 8];
        if (block_size >= 8) sm[tx] += sm[tx + 4];
        if (block_size >= 4) sm[tx] += sm[tx + 2];
        if (block_size >= 2) sm[tx] += sm[tx + 1];
    }
    if (tx == 0) ret[blockIdx.x] = sm[0];
}

float sum(Array<float, DEVICE>::Ptr darray) {
    Array<float, DEVICE>::Data data = darray->cdata();

    uint num_threads = cacc::divup(data.num_values, std::log2(data.num_values));
    uint num_blocks = cacc::divup(num_threads, REDUCTION_BLOCK_SIZE);
    cacc::Array<float, cacc::DEVICE> dtmp(num_blocks);

    dim3 grid(num_blocks);
    dim3 block(REDUCTION_BLOCK_SIZE);
    sum<REDUCTION_BLOCK_SIZE><<<grid, block>>>(
        data.data_ptr, dtmp.cdata().data_ptr, data.num_values
    );
    sum<REDUCTION_BLOCK_SIZE><<<dim3(1), block>>>(
        dtmp.cdata().data_ptr, dtmp.cdata().data_ptr, num_blocks
    );

    cacc::Array<float, cacc::HOST> tmp(dtmp);
    return tmp.cdata().data_ptr[0];
}

CACC_NAMESPACE_END

#endif /* CACC_REDUCTION_HEADER */
