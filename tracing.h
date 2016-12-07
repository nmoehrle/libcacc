/*
 * Copyright (C) 2015, Nils Moehrle
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef CACC_TRACING_HEADER
#define CACC_TRACING_HEADER

#include "bvh_tree.h"

#include "primitives.h"

CACC_NAMESPACE_BEGIN

#define TRACING_NAMESPACE_BEGIN namespace tracing {
#define TRACING_NAMESPACE_END }

TRACING_NAMESPACE_BEGIN

//WARNING works only with 1D/2D blocks
//TRACING_SSTACK_SIZE * TRACING_BLOCK_SIZE * sizeof(uint) <! block smem limit
#ifndef TRACING_BLOCK_SIZE
    #define TRACING_BLOCK_SIZE 128
#endif
#ifndef TRACING_SSTACK_SIZE
    #define TRACING_SSTACK_SIZE 12
#endif
#ifndef TRACING_GSTACK_SIZE
    #define TRACING_GSTACK_SIZE 64
#endif

constexpr uint NAI = (uint) -1;

__device__
bool trace(cacc::BVHTree<cacc::DEVICE>::Accessor const & bvh_tree,
    cacc::Ray const & ray, uint * hit_face_id_ptr = nullptr)
{
#if TRACING_SSTACK_SIZE
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int id = ty * blockDim.x + tx;
    uint __shared__ sstack[TRACING_SSTACK_SIZE * TRACING_BLOCK_SIZE];
#endif

    float t = inf;
    uint hit_face_id = NAI;
    uint gstack[TRACING_GSTACK_SIZE];
    uint node_idx = 0;

    int stack_idx = -1;
    while (true) {
        BVHTree<DEVICE>::Node node;
        node = bvh_tree.load_node(node_idx);

        if (node.left != NAI && node.right != NAI) {
            float tmin_left, tmin_right;
            AABB aabb_left = bvh_tree.load_aabb(node.left);
            bool left = intersect(ray, aabb_left, &tmin_left);
            AABB aabb_right = bvh_tree.load_aabb(node.right);
            bool right = intersect(ray, aabb_right, &tmin_right);

            if (left && right) {
                uint other;
                if (tmin_left < tmin_right) {
                    other = node.right;
                    node_idx = node.left;
                } else {
                    other = node.left;
                    node_idx = node.right;
                }
#if TRACING_SSTACK_SIZE
                if (++stack_idx < TRACING_SSTACK_SIZE)
                    sstack[TRACING_BLOCK_SIZE * stack_idx + id] = other;
                else gstack[stack_idx - TRACING_SSTACK_SIZE] = other;
#else
                gstack[++stack_idx] = other;
#endif
            } else {
                if (right) node_idx = node.right;
                if (left) node_idx = node.left;
            }
            if (!left && !right) {
                if (stack_idx < 0) break;
#if TRACING_SSTACK_SIZE
                if (stack_idx < TRACING_SSTACK_SIZE)
                    node_idx = sstack[TRACING_BLOCK_SIZE * stack_idx-- + id];
                else node_idx = gstack[stack_idx-- - TRACING_SSTACK_SIZE];
#else
                node_idx = gstack[stack_idx--];
#endif
            }
        } else {
            for (uint i = node.first; i < node.last; ++i) {
                Tri tri = bvh_tree.load_tri(i);
                if (intersect(ray, tri, &t)) {
                    hit_face_id = bvh_tree.indices_ptr[i];
                }
            }
            if (stack_idx < 0) break;
#if TRACING_SSTACK_SIZE
            if (stack_idx < TRACING_SSTACK_SIZE)
                node_idx = sstack[TRACING_BLOCK_SIZE * stack_idx-- + id];
            else node_idx = gstack[stack_idx-- - TRACING_SSTACK_SIZE];
#else
            node_idx = gstack[stack_idx--];
#endif
        }
    }

    if (hit_face_id_ptr != nullptr) *hit_face_id_ptr = hit_face_id;

    return hit_face_id != NAI;
}

TRACING_NAMESPACE_END

CACC_NAMESPACE_END

#endif /* CACC_TRACING_HEADER */
