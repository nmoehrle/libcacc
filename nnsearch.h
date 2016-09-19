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

void bind_textures(typename cacc::KDTree<3u, cacc::DEVICE>::Data const kd_tree);

template <uint K>
__device__
uint find_nns(typename cacc::KDTree<K, cacc::DEVICE>::Data const kd_tree,
    typename cacc::KDTree<K, cacc::DEVICE>::Vertex vertex,
    uint * idxs_ptr, float * dists, uint n, uint const stride = 1u);

NNSEARCH_NAMESPACE_END

CACC_NAMESPACE_END

#endif /* CACC_NNSEARCH_HEADER */
