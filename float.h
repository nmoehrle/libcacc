/*
 * Copyright (C) 2015, Nils Moehrle
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef CACC_FLOAT_HEADER
#define CACC_FLOAT_HEADER

#include "defines.h"

CACC_NAMESPACE_BEGIN

struct __align__(16) Float4 {
    float4 data;

    __forceinline__ __host__ __device__
    float const & operator[] (int i) const {
        switch (i) {
            case 0: return data.x;
            case 1: return data.y;
            case 2: return data.z;
            default: return data.w;
        }
    }

    __forceinline__ __host__ __device__
    float & operator[] (int i) {
        switch (i) {
            case 0: return data.x;
            case 1: return data.y;
            case 2: return data.z;
            default: return data.w;
        }
    }
};

struct __align__(8) Float2 {
    float2 data;
    
    __forceinline__ __host__ __device__
    float const & operator[] (int i) const {
        return i ? data.y : data.x;
    }
    
    __forceinline__ __host__ __device__
    float & operator[] (int i) {
        return i ? data.y : data.x;
    }
};

CACC_NAMESPACE_END

#endif /* CACC_FLOAT_HEADER */
