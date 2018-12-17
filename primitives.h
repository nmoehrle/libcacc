/*
 * Copyright (C) 2015-2018, Nils Moehrle, Patrick Seemann
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef CACC_PRIMITIVES_HEADER
#define CACC_PRIMITIVES_HEADER

#include "acc/primitives.h"

#include "defines.h"

#include "vector.h"

constexpr float inf = std::numeric_limits<float>::infinity();
constexpr float eps = std::numeric_limits<float>::epsilon();

CACC_NAMESPACE_BEGIN

typedef acc::AABB<cacc::Vec3f> AABB;
typedef acc::Tri<cacc::Vec3f> Tri;

#pragma __align__(16)
struct Ray {
    cacc::Vec3f origin;
    cacc::Vec3f dir;

    __device__ __forceinline__
    float get_tmax() const { return dir[3]; }

    __device__ __forceinline__
    void set_tmax(float v) { dir[3] = v; }

    __device__ __forceinline__
    float get_tmin() const { return origin[3]; }

    __device__ __forceinline__
    void set_tmin(float v) { origin[3] = v; }
};

/* Derived form Tavian Barnes implementation posted in
 * http://tavianator.com/fast-branchless-raybounding-box-intersections-part-2-nans/
 * on 23rd March 2015 */
__device__ __forceinline__ bool
intersect(Ray const & ray, AABB const & aabb, float * tmin_ptr) {
    float tmin = ray.get_tmin();
    float tmax = ray.get_tmax();
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        float t1 = (aabb.min[i] - ray.origin[i]) / ray.dir[i];
        float t2 = (aabb.max[i] - ray.origin[i]) / ray.dir[i];

        tmin = max(tmin, min(min(t1, t2), inf));
        tmax = min(tmax, max(max(t1, t2), -inf));
    }
    *tmin_ptr = tmin;
    return tmax >= max(tmin, 0.0f);
}

__device__ __forceinline__ bool
intersect(Ray const & ray, Tri const & tri, float * t_ptr) {
    cacc::Vec3f v0 = tri.b - tri.a;
    cacc::Vec3f v1 = tri.c - tri.a;
    cacc::Vec3f normal = cross(v1, v0);
    if (norm(normal) < eps) return false;
    normal.normalize();

    float cosine = dot(normal, ray.dir);
    if (std::abs(cosine) < eps) return false;

    float t = -dot(normal, ray.origin - tri.a) / cosine;

    if (t < ray.get_tmin() || ray.get_tmax() < t || t > *t_ptr) return false;
    cacc::Vec3f v2 = (ray.origin - tri.a) + t * ray.dir;

    /* Derived from the book "Real-Time Collision Detection"
     * by Christer Ericson published by Morgan Kaufmann in 2005 */
    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);
    float denom = d00 * d11 - d01 * d01;

    float b1 = (d11 * d20 - d01 * d21) / denom;
    float b2 = (d00 * d21 - d01 * d20) / denom;
    float b0 = 1.0f - b1 - b2;

    constexpr float eps = 1e-3f;
    if (-eps > b0 || b0 > 1.0f + eps) return false;
    if (-eps > b1 || b1 > 1.0f + eps) return false;
    if (-eps > b2 || b2 > 1.0f + eps) return false;

    *t_ptr = t;
    return true;
}

CACC_NAMESPACE_END

#endif /* CACC_PRIMITIVES_HEADER */
