/*
 * Copyright (C) 2015, Nils Moehrle
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include "nnsearch.h"

static texture<uint4, 1> nodes;
static texture<float4, 1> verts;

CACC_NAMESPACE_BEGIN

NNSEARCH_NAMESPACE_BEGIN

void bind_textures(typename KDTree<3u, DEVICE>::Data const kd_tree) {
    static_assert(sizeof(KDTree<3u, DEVICE>::Node) == sizeof(uint4), "");
    CHECK(cudaBindTexture(NULL, nodes, kd_tree.nodes_ptr,
        kd_tree.num_nodes * sizeof(KDTree<3u, DEVICE>::Node)));
    static_assert(sizeof(KDTree<3u, DEVICE>::Vertex) == sizeof(float4), "");
    CHECK(cudaBindTexture(NULL, verts, kd_tree.verts_ptr,
        kd_tree.num_verts * sizeof(float4)));
}

__device__ __forceinline__
typename KDTree<3u, DEVICE>::Node load_node(uint idx) {
    typename KDTree<3u, DEVICE>::Node node;
    node.rldv = tex1Dfetch(nodes, idx);
    return node;
}

__device__ __forceinline__
typename KDTree<3u, DEVICE>::Vertex load_vertex(uint idx) {
    float4 v = tex1Dfetch(verts, idx);
    typename KDTree<3u, DEVICE>::Vertex vertex(v.x, v.y, v.z);
    return vertex;
}

__device__ __forceinline__
uint
max_element(float const * values, uint n, uint const stride)
{
    float max = 0;
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
    float max = 0;
    for (int i = 0; i < n; ++i) {
        float v = values[i * stride];
        if (v > max) {
            max = v;
        }
    }
    return max;
}

template
__device__
uint find_nns<3u>(cacc::KDTree<3u, cacc::DEVICE>::Data const kd_tree,
    cacc::KDTree<3u, cacc::DEVICE>::Vertex vertex,
    uint * idxs_ptr, float * dists, uint n, uint const stride);

constexpr float inf = std::numeric_limits<float>::infinity();

template <uint K>
__device__
uint find_nns(typename cacc::KDTree<K, cacc::DEVICE>::Data const kd_tree,
    typename cacc::KDTree<K, cacc::DEVICE>::Vertex vertex,
    uint * idxs_ptr, float * dists_ptr, uint k, uint const stride)
{
    const int tx = threadIdx.x;

    uint n = 0;
    float max_dist = inf;
    uint node_idx = 0;
    bool down = true;
    uint gstack[NNSEARCH_GSTACK_SIZE];
    uint __shared__ sstack[NNSEARCH_SSTACK_SIZE * NNSEARCH_BLOCK_SIZE];

    int stack_idx = -1;
    while (true) {
        typename KDTree<K, DEVICE>::Node node = load_node(node_idx);
        typename KDTree<K, DEVICE>::Vertex vert = load_vertex(node.vid);

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
                    if (++stack_idx < NNSEARCH_SSTACK_SIZE) sstack[NNSEARCH_BLOCK_SIZE * stack_idx + tx] = node_idx;
                    else gstack[stack_idx] = node_idx;
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
            if (stack_idx < NNSEARCH_SSTACK_SIZE) node_idx = sstack[NNSEARCH_BLOCK_SIZE * stack_idx-- + tx];
            else node_idx = gstack[stack_idx--];
        }
    }

    return n;
}

NNSEARCH_NAMESPACE_END

CACC_NAMESPACE_END
