/*
 * Copyright (C) 2015-2018, Nils Moehrle
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include "kd_tree.h"

#include <iostream>

ACC_NAMESPACE_BEGIN

template <> template <>
cacc::KDTree<3, cacc::HOST> KDTree<3, unsigned>::convert<cacc::KDTree<3, cacc::HOST> >(
    KDTree<3, unsigned> const & kd_tree) {

    uint num_nodes = kd_tree.num_nodes;
    uint num_verts = kd_tree.num_nodes;
    cacc::KDTree<3, cacc::HOST> nkd_tree(num_nodes, num_verts);
    cacc::KDTree<3, cacc::HOST>::Data & data = nkd_tree.data;
    for (std::size_t i = 0; i < num_nodes; ++i) {
        cacc::KDTree<3, cacc::HOST>::Node & nnode = data.nodes_ptr[i];
        KDTree<3, unsigned>::Node const & node = kd_tree.nodes[i];
        nnode.vid = node.vertex_id;
        nnode.dim = node.d;
        nnode.left = node.left;
        nnode.right = node.right;
    }
    for (std::size_t i = 0; i < num_verts; ++i) {
        cacc::KDTree<3, cacc::HOST>::Vertex vert(*kd_tree.vertices[i]);
        data.verts_ptr[i] = vert;
    }
    return nkd_tree;
}

ACC_NAMESPACE_END

CACC_NAMESPACE_BEGIN

template <> template <>
KDTree<3, DEVICE>::KDTree(acc::KDTree<3, unsigned> const & kd_tree) : KDTree() {
    *this = acc::KDTree<3, unsigned>::convert<KDTree<3, HOST> >(kd_tree);
}

CACC_NAMESPACE_END
