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

//SSTACK_SIZE * NNSEARCH_BLOCK_SIZE has to be less than smem
#define NNSEARCH_BLOCK_SIZE 128
#define SSTACK_SIZE 6
#define GSTACK_SIZE 64

constexpr uint NAI = (uint) -1;

template <uint K>
void bind_textures(typename cacc::KDTree<K, cacc::DEVICE>::Data const kd_tree);

template <uint K>
__device__
void find_nns(typename cacc::KDTree<K, cacc::DEVICE>::Data const kd_tree,
    typename cacc::KDTree<K, cacc::DEVICE>::Vertex vertex,
    uint * idxs_ptr, float * dists, uint * n_ptr, uint const stride = 1u);

NNSEARCH_NAMESPACE_END

CACC_NAMESPACE_END

#endif /* CACC_NNSEARCH_HEADER */
