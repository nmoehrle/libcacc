/*
 * Copyright (C) 2015, Nils Moehrle
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef CACC_NNSEARCH_HEADER
#define CACC_NNSEARCH_HEADER

#include "kd_tree.h"

CACC_NAMESPACE_BEGIN

#define NNSEARCH_NAMESPACE_BEGIN namespace nnsearch {
#define NNSEARCH_NAMESPACE_END }

NNSEARCH_NAMESPACE_BEGIN

//NNSEARCH_SSTACK_SIZE * NNSEARCH_BLOCK_SIZE has to be less than smem
#ifndef NNSEARCH_BLOCK_SIZE
    #define NNSEARCH_BLOCK_SIZE 128
#endif
#ifndef NNSEARCH_SSTACK_SIZE
    #define NNSEARCH_SSTACK_SIZE 6
#endif
#ifndef NNSEARCH_GSTACK_SIZE
    #define NNSEARCH_GSTACK_SIZE 64
#endif

constexpr uint NAI = (uint) -1;

constexpr float lowest = std::numeric_limits<float>::lowest();
constexpr float inf = std::numeric_limits<float>::infinity();

__device__ __forceinline__
uint
max_element(float const * values, uint n, uint const stride)
{
    float max = lowest;
    uint ret = (uint) -1;
    for (int i = 0; i < n; ++i) {
        float v = values[i * stride];
        if (v > max) {
            max = v;
            ret = i;
        }
    }
    return ret;
}

__device__ __forceinline__
float
max_value(float const * values, uint n, uint const stride)
{
    float max = lowest;
    for (int i = 0; i < n; ++i) {
        float v = values[i * stride];
        if (v > max) {
            max = v;
        }
    }
    return max;
}

template <uint K>
__device__
bool find_nn(typename cacc::KDTree<K, cacc::DEVICE>::Accessor const & kd_tree,
    typename cacc::KDTree<K, cacc::DEVICE>::Vertex vertex,
    uint * idx_ptr, float * dist_ptr)
{
#if NNSEARCH_SSTACK_SIZE
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int id = ty * blockDim.x + tx;
    uint __shared__ sstack[NNSEARCH_SSTACK_SIZE * NNSEARCH_BLOCK_SIZE];
#endif

    uint idx = NAI;
    float max_dist = inf;

    uint node_idx = 0;
    bool down = true;
    uint gstack[NNSEARCH_GSTACK_SIZE];

    int stack_idx = -1;
    while (true) {
        typename KDTree<K, DEVICE>::Node node;
        node = kd_tree.load_node(node_idx);
        typename KDTree<K, DEVICE>::Vertex vert;
        vert = kd_tree.load_vertex(node.vid);

        float diff = vertex[node.dim] - vert[node.dim];
        if (down) {
            float dist = norm(vertex - vert);
            if (dist < max_dist) {
                idx = node.vid;
                max_dist = dist;
            }

            if (node.left != NAI || node.right != NAI) {
                /* Inner node - traverse further down. */
                down = true;

                if (node.left != NAI && node.right != NAI) {
#if NNSEARCH_SSTACK_SIZE
                    if (++stack_idx < NNSEARCH_SSTACK_SIZE)
                        sstack[NNSEARCH_BLOCK_SIZE * stack_idx + id] = node_idx;
                    else gstack[stack_idx - NNSEARCH_SSTACK_SIZE] = node_idx;
#else
                    gstack[++stack_idx] = node_idx;
#endif
                }

                float diff = vertex[node.dim] - vert[node.dim];

                uint next = (diff < 0.0f) ? node.left : node.right;
                uint other = (diff < 0.0f) ? node.right : node.left;

                node_idx = (next != NAI) ? next : other;
            } else {
                /* Leaf - traverse up and search for next node. */
                down = false;
                node_idx = NAI;
            }
        } else {
            if (std::abs(diff) < max_dist) {
                down = true;
                node_idx = (diff < 0.0f) ? node.right : node.left;
            } else {
                down = false;
                node_idx = NAI;
            }
        }

        if (node_idx == NAI) {
            if (stack_idx < 0) break;
#if NNSEARCH_SSTACK_SIZE
            if (stack_idx < NNSEARCH_SSTACK_SIZE)
                node_idx = sstack[NNSEARCH_BLOCK_SIZE * stack_idx-- + id];
            else node_idx = gstack[stack_idx-- - NNSEARCH_SSTACK_SIZE];
#else
            node_idx = gstack[stack_idx--];
#endif
        }
    }

    if (idx == NAI) return false;

    if (idx_ptr != nullptr) *idx_ptr = idx;
    if (dist_ptr != nullptr) *dist_ptr = max_dist;

    return true;
}

template <uint K>
__device__
uint find_nns(typename cacc::KDTree<K, cacc::DEVICE>::Accessor const & kd_tree,
    typename cacc::KDTree<K, cacc::DEVICE>::Vertex vertex,
    uint * idxs_ptr, float * dists_ptr, uint k, uint const stride = 1u)
{
#if NNSEARCH_SSTACK_SIZE
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int id = ty * blockDim.x + tx;
    uint __shared__ sstack[NNSEARCH_SSTACK_SIZE * NNSEARCH_BLOCK_SIZE];
#endif

    uint n = 0;
    float max_dist = inf;
    uint node_idx = 0;
    bool down = true;
    uint gstack[NNSEARCH_GSTACK_SIZE];

    int stack_idx = -1;
    while (true) {
        typename KDTree<K, DEVICE>::Node node;
        node = kd_tree.load_node(node_idx);
        typename KDTree<K, DEVICE>::Vertex vert;
        vert = kd_tree.load_vertex(node.vid);

        float diff = vertex[node.dim] - vert[node.dim];
        if (down) {
            float dist = norm(vertex - vert);
            if (dist < max_dist) {
                if (n < k) {
                    idxs_ptr[n * stride] = node.vid;
                    dists_ptr[n * stride] = dist;
                    n += 1;
                } else {
                    uint i = max_element(dists_ptr, n, stride);
                    idxs_ptr[i * stride] = node.vid;
                    dists_ptr[i * stride] = dist;
                }

                if (n == k) {
                    max_dist = max_value(dists_ptr, n, stride);
                }
            }

            if (node.left != NAI || node.right != NAI) {
                /* Inner node - traverse further down. */
                down = true;

                if (node.left != NAI && node.right != NAI) {
#if NNSEARCH_SSTACK_SIZE
                    if (++stack_idx < NNSEARCH_SSTACK_SIZE)
                        sstack[NNSEARCH_BLOCK_SIZE * stack_idx + id] = node_idx;
                    else gstack[stack_idx - NNSEARCH_SSTACK_SIZE] = node_idx;
#else
                    gstack[++stack_idx] = node_idx;
#endif
                }

                float diff = vertex[node.dim] - vert[node.dim];

                uint next = (diff < 0.0f) ? node.left : node.right;
                uint other = (diff < 0.0f) ? node.right : node.left;

                node_idx = (next != NAI) ? next : other;
            } else {
                /* Leaf - traverse up and search for next node. */
                down = false;
                node_idx = NAI;
            }
        } else {
            if (std::abs(diff) < max_dist) {
                down = true;
                node_idx = (diff < 0.0f) ? node.right : node.left;
            } else {
                down = false;
                node_idx = NAI;
            }
        }

        if (node_idx == NAI) {
            if (stack_idx < 0) break;
#if NNSEARCH_SSTACK_SIZE
            if (stack_idx < NNSEARCH_SSTACK_SIZE) node_idx = sstack[NNSEARCH_BLOCK_SIZE * stack_idx-- + id];
            else node_idx = gstack[stack_idx-- - NNSEARCH_SSTACK_SIZE];
#else
            node_idx = gstack[stack_idx--];
#endif
        }
    }

    return n;
}

NNSEARCH_NAMESPACE_END

CACC_NAMESPACE_END

#endif /* CACC_NNSEARCH_HEADER */
