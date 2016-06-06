/*
 * Copyright (C) 2015, Nils Moehrle
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef CACC_RAYS_HEADER
#define CACC_RAYS_HEADER

#include "primitives.h"
#include "vector.h"

#include "defines.h"

CACC_NAMESPACE_BEGIN

class Rays {
public:
    struct Data {
        uint num_rays;
        Ray * rays_ptr;
    };

private:
    Data data;

public:
    Rays(uint num_rays) {
        data.num_rays = num_rays;
        CHECK(cudaMalloc(&data.rays_ptr, num_rays * sizeof(Ray)));
    }

    ~Rays() {
        CHECK(cudaFree(data.rays_ptr));
    }

    Data const & cdata() {
        return data;
    }
};

CACC_NAMESPACE_END

#endif /* CACC_RAYS_HEADER */
