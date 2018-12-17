/*
 * Copyright (C) 2015-2018, Nils Moehrle
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef CACC_GRAPHICSRESOURCE_HEADER
#define CACC_GRAPHICSRESOURCE_HEADER

#include "defines.h"

#include <cuda_runtime_api.h>

CACC_NAMESPACE_BEGIN

class GraphicsResource {
private:
    cudaGraphicsResource_t res;
public:
    GraphicsResource() : res(nullptr) {}
    ~GraphicsResource() {
        if (res) CHECK(cudaGraphicsUnregisterResource(res));
    }
    cudaGraphicsResource_t & ptr() { return res; }
};

CACC_NAMESPACE_END

#endif /* CACC_GRAPHICSRESOURCE_HEADER */
