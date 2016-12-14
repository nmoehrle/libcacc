/*
 * Copyright (C) 2015, Nils Moehrle
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef CACC_MATH_HEADER
#define CACC_MATH_HEADER

#include <cassert>

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
    static_assert(sizeof(float) == sizeof(uint32_t), "");
    typedef union {
        float f;
        uint32_t u;
    } Alias;

    Alias tmp = {f};
    uint32_t mask = -int32_t(tmp.u >> 31) | 0x80000000;
    return tmp.u ^ mask;
}

__forceinline__ __host__ __device__
float uint32_to_float(uint32_t u)
{
    static_assert(sizeof(uint32_t) == sizeof(float), "");
    typedef union {
        uint32_t u;
        float f;
    } Alias;

    uint32_t mask = ((u >> 31) - 1) | 0x80000000;
    Alias tmp = {u ^ mask};
    return tmp.f;
}

__forceinline__ __host__ __device__
float float_to_uint32_as_float(float f)
{
    static_assert(sizeof(float) == sizeof(uint32_t), "");
    typedef union {
        float f;
        uint32_t u;
    } Alias;

    Alias tmp;
    tmp.u = float_to_uint32(f);
    return tmp.f;
}

__forceinline__ __host__ __device__
float uint32_as_float_to_float(float f)
{
    static_assert(sizeof(float) == sizeof(uint32_t), "");
    typedef union {
        float f;
        uint32_t u;
    } Alias;

    Alias tmp;
    tmp.f = f;
    return uint32_to_float(tmp.u);
}

CACC_NAMESPACE_END

#endif /* CACC_MATH_HEADER */
