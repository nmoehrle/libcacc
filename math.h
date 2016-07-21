/*
 * Copyright (C) 2015, Nils Moehrle
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef CACC_MATH_HEADER
#define CACC_MATH_HEADER

#include "defines.h"

CACC_NAMESPACE_BEGIN

inline
uint divup(uint a, uint b) {
    return a / b  + (a % b != 0);
}

/* Derived from http://stereopsis.com/radix.html */
__forceinline__ __host__ __device__
uint32_t float_to_uint32(float f)
{
    uint32_t v = reinterpret_cast<uint32_t&>(f);
    uint32_t mask = -int32_t(v >> 31) | 0x80000000;
    return v ^ mask;
}

__forceinline__ __host__ __device__
float uint32_to_float(uint32_t v)
{
    uint32_t mask = ((v >> 31) - 1) | 0x80000000;
    uint32_t f = v ^ mask;
    return reinterpret_cast<float&>(f);
}

CACC_NAMESPACE_END

#endif /* CACC_MATH_HEADER */
