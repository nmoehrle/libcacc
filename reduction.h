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

#define REDUCTION_NAMESPACE_BEGIN namespace reduction {
#define REDUCTION_NAMESPACE_END }

REDUCTION_NAMESPACE_BEGIN

struct FAdd {
    __device__ __forceinline__
    float operator()(float lhs, float rhs) {
        return lhs + rhs;
    }

    __device__ __forceinline__
    void operator()(float *addr, float val) {
        atomicAdd(addr, val);
    }
};

struct FMin {
    __device__ __forceinline__
    float operator()(float lhs, float rhs) {
        return min(lhs, rhs);
    }

    __device__ __forceinline__
    void operator()(float *addr, float val) {
        atomicMin((uint32_t *)addr, float_to_uint32(val));
    }
};

struct FMax {
    __device__ __forceinline__
    float operator()(float lhs, float rhs) {
        return max(lhs, rhs);
    }

    __device__ __forceinline__
    void operator()(float *addr, float val) {
        atomicMax((uint32_t *)addr, float_to_uint32(val));
    }
};

template <typename F>
float convert_to_gmem_repr(float val) {
    return val;
}

template <typename F>
float convert_to_host_repr(float val) {
    return val;
}

template <>
float convert_to_gmem_repr<FMin>(float val) {
    return float_to_uint32_as_float(val);
}

template <>
float convert_to_gmem_repr<FMax>(float val) {
    return float_to_uint32_as_float(val);
}

template <>
float convert_to_host_repr<FMin>(float val) {
    return uint32_as_float_to_float(val);
}

template <>
float convert_to_host_repr<FMax>(float val) {
    return uint32_as_float_to_float(val);
}

template <typename F>
__inline__ __device__
float reduce_warp(float val) {
    #pragma unroll
    for (int stride = warpSize >> 1; stride > 0; stride >>= 1) {
        val = F()(val, __shfl_down(val, stride));
    }
    return val;
}

template <typename F>
__global__
void reduce_kernel(float * values, float * ret, uint n, float init)
{
    __shared__ float sm[32];
    uint lane = threadIdx.x % warpSize;
    uint tx = threadIdx.x;
    uint wx = threadIdx.x / warpSize;
    uint gstride = blockDim.x * gridDim.x;

    float val = init;
    for (uint i = blockIdx.x * blockDim.x + tx; i < n; i += gstride) {
        val = F()(val, values[i]);
    }

    val = reduce_warp<F>(val);

    if (lane == 0) sm[wx] = val;

    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? sm[lane] : init;

    if (wx == 0) val = reduce_warp<F>(val);

    if (tx == 0) F()(ret, val);
}

template <typename F>
float reduce(Array<float, DEVICE>::Ptr darray, float init) {
    Array<float, DEVICE>::Data data = darray->cdata();

    uint num_threads = cacc::divup(data.num_values, std::log2(data.num_values));
    uint num_blocks = cacc::divup(num_threads, REDUCTION_BLOCK_SIZE);

    Variable<float, DEVICE> dret(convert_to_gmem_repr<F>(init));

    dim3 grid(num_blocks);
    dim3 block(REDUCTION_BLOCK_SIZE);
    reduce_kernel<F><<<grid, block>>>(
        data.data_ptr, dret.cptr(), data.num_values, init
    );

    Variable<float, HOST> ret(dret);

    return convert_to_host_repr<F>(ret.ref());
}

float sum(Array<float, DEVICE>::Ptr darray) {
    return reduce<FAdd>(darray, 0.0f);
}

float min(Array<float, DEVICE>::Ptr darray) {
    return reduce<FMin>(darray, std::numeric_limits<float>::max());
}

float max(Array<float, DEVICE>::Ptr darray) {
    return reduce<FMax>(darray, std::numeric_limits<float>::lowest());
}

REDUCTION_NAMESPACE_END

CACC_NAMESPACE_END

#endif /* CACC_REDUCTION_HEADER */
