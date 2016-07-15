/*
 * Copyright (C) 2015, Nils Moehrle
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include "nnsearch.h"

texture<uint4, 1> nodes;
texture<float2, 1> verts_2d;
texture<float4, 1> verts_4d;

CACC_NAMESPACE_BEGIN

NNSEARCH_NAMESPACE_BEGIN

template <uint K>
void bind_textures(typename KDTree<K, DEVICE>::Data const kd_tree) {
    static_assert(sizeof(KDTree<K, DEVICE>::Node) == sizeof(uint4), "");
    CHECK(cudaBindTexture(NULL, nodes, kd_tree.nodes_ptr,
        kd_tree.num_nodes * sizeof(KDTree<K, DEVICE>::Node)));
    if (K <= 2) {
        static_assert(sizeof(KDTree<K, DEVICE>::Vertex) == sizeof(float2), "");
        CHECK(cudaBindTexture(NULL, verts_2d, kd_tree.verts_ptr,
            kd_tree.num_verts * sizeof(float2)));
    } else {
        static_assert(sizeof(KDTree<K, DEVICE>::Vertex) == sizeof(float4), "");
        CHECK(cudaBindTexture(NULL, verts_4d, kd_tree.verts_ptr,
            kd_tree.num_verts * sizeof(float4)));
    }
}

template <uint K>
__device__ __forceinline__
typename KDTree<K, DEVICE>::Node load_node(uint idx) {
    typename KDTree<K, DEVICE>::Node node;
    node.rldv = tex1Dfetch(nodes, idx);
    return node;
}

template <uint K>
__device__ __forceinline__
typename KDTree<K, DEVICE>::Vertex load_vertex(uint idx) {
    typename KDTree<K, DEVICE>::Vertex vertex;
    if (K <= 2) {
        vertex = tex1Dfetch(verts_2d, idx);
    } else {
        vertex = tex1Dfetch(verts_4d, idx);
    }
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

template <uint K>
__device__
void find_nns(typename cacc::KDTree<K, cacc::DEVICE>::Data const kd_tree,
    typename cacc::KDTree<K, cacc::DEVICE>::Vertex vertex,
    uint * idxs_ptr, float * dists_ptr, uint * n_ptr, uint const stride)
{
    const int tx = threadIdx.x;

    uint n = 0;
    uint k = *n_ptr;
    float max_dist = std::numeric_limits<float>::infinity();
    uint node_idx = 0;
    bool down = true;
    uint gstack[GSTACK_SIZE];
    uint __shared__ sstack[SSTACK_SIZE * NNSEARCH_BLOCK_SIZE];

    int stack_idx = -1;
    while (true) {
        typename KDTree<K, DEVICE>::Node node = load_node<K>(node_idx);
        typename KDTree<K, DEVICE>::Vertex vert = load_vertex<K>(node.vid);

        float diff = vertex[node.dim] - vert[node.dim];
        if (down) {
            float dist = norm(vertex, vert);
            if (dist < max_dist) {
                if (n < k) {
                    idxs_ptr[n] = node.vid;
                    dists_ptr[n] = node.dist;
                    n += 1;
                } else {
                    uint i = max_element(dists_ptr, n, stride);
                    idxs_ptr[i * stride] = node.vid;
                    dists_ptr[i * stride] = dist;
                    max_dist = max_value(dists_ptr, n, stride);
                }
            }

            if (node.left == NAI && node.right == NAI) {
                node_idx = NAI;
            } else {
                down = true;
                uint other;

                if (diff < 0.0f) {
                    node_idx = node.left;
                    other = node.right;
                } else {
                    node_idx = node.right;
                    other = node.left;
                }

                if (node_idx != NAI) {
                    if (++stack_idx < SSTACK_SIZE) sstack[SSTACK_SIZE * tx + stack_idx] = other;
                    else gstack[stack_idx] = other;

                } else {
                    node_idx = other;
                }
            }
        } else {
            if (std::abs(diff) >= max_dist) {
                node_idx = NAI;
            } else {
                down = true;

                if (diff < 0.0f) {
                    node_idx = node.right;
                } else {
                    node_idx = node.left;
                }
            }
        }

        if (node_idx == NAI) {
            if (stack_idx < 0) break;
            if (stack_idx < SSTACK_SIZE) node_idx = sstack[SSTACK_SIZE * tx + stack_idx--];
            else node_idx = gstack[stack_idx--];
            down = false;
        }
    }

    *n_ptr = n;
}

NNSEARCH_NAMESPACE_END

CACC_NAMESPACE_END
