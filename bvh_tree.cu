#include "bvh_tree.h"

#include <iostream>

ACC_NAMESPACE_BEGIN

template <> template <>
cacc::BVHTree<cacc::HOST> BVHTree<unsigned int, math::Vec3f>::convert<cacc::BVHTree<cacc::HOST> >(
        BVHTree<unsigned int, math::Vec3f> const & bvh_tree) {

    uint num_nodes = bvh_tree.num_nodes;
    uint num_faces = bvh_tree.tris.size();
    cacc::BVHTree<cacc::HOST> nbvh_tree(num_nodes, num_faces);
    cacc::BVHTree<cacc::HOST>::Data & data = nbvh_tree.data;
    for (std::size_t i = 0; i < num_nodes; ++i) {
        cacc::BVHTree<cacc::HOST>::Node & nnode = data.nodes_ptr[i];
        BVHTree<unsigned int, math::Vec3f>::Node const & node = bvh_tree.nodes[i];
        nnode.first = node.first;
        nnode.last = node.last;
        nnode.left = node.left;
        nnode.right = node.right;
        cacc::BVHTree<cacc::HOST>::AABB & naabb = data.aabbs_ptr[i];
        naabb.min = cacc::Vec3f(*node.aabb.min);
        naabb.max = cacc::Vec3f(*node.aabb.max);
    }
    for (std::size_t i = 0; i < num_faces; ++i) {
        data.indices_ptr[i] = bvh_tree.indices[i];

        cacc::BVHTree<cacc::HOST>::Tri & ntri = data.tris_ptr[i];
        BVHTree<unsigned int, math::Vec3f>::Tri const & tri = bvh_tree.tris[i];
        ntri.a = cacc::Vec3f(*tri.a);
        ntri.b = cacc::Vec3f(*tri.b);
        ntri.c = cacc::Vec3f(*tri.c);
    }
    return nbvh_tree;
}

ACC_NAMESPACE_END

CACC_NAMESPACE_BEGIN

template <> template <>
BVHTree<DEVICE>::BVHTree(acc::BVHTree<unsigned int, math::Vec3f> const & bvh_tree) : BVHTree() {
    *this = acc::BVHTree<unsigned int, math::Vec3f>::convert<BVHTree<HOST> >(bvh_tree);
}

CACC_NAMESPACE_END
