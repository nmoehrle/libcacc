/*
 * Copyright (C) 2015-2018, Nils Moehrle
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef CACC_DEFINES_HEADER
#define CACC_DEFINES_HEADER

#define CACC_NAMESPACE_BEGIN namespace cacc {
#define CACC_NAMESPACE_END }

#include <iostream>
#include <cuda_runtime.h>

CACC_NAMESPACE_BEGIN

#define CHECK(CALL) \
do { \
    cudaError_t err = (CALL); \
    if (cudaSuccess != err) { \
        std::cerr << "CUDA error in " \
            << __FILE__ << ":" << __LINE__ << " (" << #CALL << "): " \
            << cudaGetErrorString(err) << " (" << err << ")" << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

enum Location {
    HOST,
    DEVICE
};

CACC_NAMESPACE_END

#endif /* CACC_DEFINES_HEADER */
