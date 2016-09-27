/*
 * Copyright (C) 2015, Nils Moehrle
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include "tracing.h"

#include "primitives.h"

static texture<uint4, 1> nodes;
static texture<float4, 1> aabbs;
static texture<float4, 1> tris;

CACC_NAMESPACE_BEGIN

TRACING_NAMESPACE_BEGIN

void bind_textures(BVHTree<DEVICE>::Data const bvh_tree) {
    static_assert(sizeof(BVHTree<DEVICE>::Node) == sizeof(uint4), "");
    static_assert(sizeof(AABB) == 2 * sizeof(float4), "");
    static_assert(sizeof(Tri) == 3 * sizeof(float4), "");
    CHECK(cudaBindTexture(NULL, nodes, bvh_tree.nodes_ptr,
        bvh_tree.num_nodes * sizeof(BVHTree<DEVICE>::Node)));
    CHECK(cudaBindTexture(NULL, aabbs, bvh_tree.aabbs_ptr,
        bvh_tree.num_nodes * 2 * sizeof(float4)));
    CHECK(cudaBindTexture(NULL, tris, bvh_tree.tris_ptr,
        bvh_tree.num_faces * 3 * sizeof(float4)));
}

__device__ __forceinline__
BVHTree<DEVICE>::Node load_node(uint idx) {
    BVHTree<DEVICE>::Node node;
    node.rllf = tex1Dfetch(nodes, idx);
    return node;
}

__device__ __forceinline__
AABB load_aabb(uint idx) {
    AABB aabb;
    float4 min = tex1Dfetch(aabbs, 2 * idx + 0);
    aabb.min = Vec3f(min.x, min.y, min.z);
    float4 max = tex1Dfetch(aabbs, 2 * idx + 1);
    aabb.max = Vec3f(max.x, max.y, max.z);
    return aabb;
}

__device__ __forceinline__
Tri load_tri(uint idx) {
    Tri tri;
    float4 a = tex1Dfetch(tris, 3 * idx + 0);
    tri.a = Vec3f(a.x, a.y, a.z);
    float4 b = tex1Dfetch(tris, 3 * idx + 1);
    tri.b = Vec3f(b.x, b.y, b.z);
    float4 c = tex1Dfetch(tris, 3 * idx + 2);
    tri.c = Vec3f(c.x, c.y, c.z);
    return tri;
}

__device__
bool trace(BVHTree<DEVICE>::Data const bvh_tree,
    Ray const ray, uint * hit_face_id_ptr)
{
    const int tx = threadIdx.x;

    float t = inf;
    uint hit_face_id = NAI;
    uint gstack[TRACING_GSTACK_SIZE];
    uint __shared__ sstack[TRACING_SSTACK_SIZE * TRACING_BLOCK_SIZE];
    uint node_idx = 0;

    int stack_idx = -1;
    while (true) {
        BVHTree<DEVICE>::Node node;
        node = load_node(node_idx);

        if (node.left != NAI && node.right != NAI) {
            float tmin_left, tmin_right;
            AABB aabb_left = load_aabb(node.left);
            bool left = intersect(ray, aabb_left, &tmin_left);
            AABB aabb_right = load_aabb(node.right);
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
                if (++stack_idx < TRACING_SSTACK_SIZE) sstack[TRACING_BLOCK_SIZE * stack_idx + tx] = other;
                else gstack[stack_idx] = other;
            } else {
                if (right) node_idx = node.right;
                if (left) node_idx = node.left;
            }
            if (!left && !right) {
                if (stack_idx < 0) break;
                if (stack_idx < TRACING_SSTACK_SIZE) node_idx = sstack[TRACING_BLOCK_SIZE * stack_idx-- + tx];
                else node_idx = gstack[stack_idx--];
            }
        } else {
            for (uint i = node.first; i < node.last; ++i) {
                Tri tri = load_tri(i);
                if (intersect(ray, tri, &t)) {
                    hit_face_id = bvh_tree.indices_ptr[i];
                }
            }
            if (stack_idx < 0) break;
            if (stack_idx < TRACING_SSTACK_SIZE) node_idx = sstack[TRACING_BLOCK_SIZE * stack_idx-- + tx];
            else node_idx = gstack[stack_idx--];
        }
    }

    if (hit_face_id_ptr != nullptr) *hit_face_id_ptr = hit_face_id;

    return hit_face_id != NAI;
}

TRACING_NAMESPACE_END

CACC_NAMESPACE_END
