/*
 * Copyright (C) 2015, Nils Moehrle
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef CACC_BVHTREE_HEADER
#define CACC_BVHTREE_HEADER

#include "acc/bvh_tree.h"
#include "vector.h"

#include "primitives.h"

#include "defines.h"


#include <cuda_runtime.h>

CACC_NAMESPACE_BEGIN

template <Location L>
class BVHTree {
public:
    typedef std::shared_ptr<BVHTree> Ptr;
    const uint NAI = std::numeric_limits<uint>::max();

    typedef cacc::Tri Tri;
    typedef cacc::AABB AABB;

    #pragma __align__(64)
    struct Node {
        union {
            struct {
                uint first;
                uint last;
                uint left;
                uint right;
            };
            uint4 rllf;
        };
    };

    struct Data {
        uint num_nodes;
        uint num_faces;
        Node * nodes_ptr;
        AABB * aabbs_ptr;
        Tri * tris_ptr;
        uint * indices_ptr;
    };

private:
    Data data;

    void init(uint num_nodes, uint num_faces) {
        data.num_nodes = num_nodes;
        data.num_faces = num_faces;
        if (L == HOST) {
            data.nodes_ptr = new Node[num_nodes];
            data.aabbs_ptr = new AABB[num_nodes];
            data.tris_ptr = new Tri[num_faces];
            data.indices_ptr = new uint[num_faces];
        } else {
            CHECK(cudaMalloc(&data.nodes_ptr, num_nodes * sizeof(Node)));
            CHECK(cudaMalloc(&data.aabbs_ptr, num_nodes * sizeof(AABB)));
            CHECK(cudaMalloc(&data.tris_ptr, num_faces * sizeof(Tri)));
            CHECK(cudaMalloc(&data.indices_ptr, num_faces * sizeof(uint)));
        }
    }

    template <typename T>
    void cleanup(T * ptr) {
        if (ptr == nullptr) return;
        if (L == HOST) delete[] ptr;
        if (L == DEVICE) CHECK(cudaFree(ptr));
        ptr = nullptr;
    }

    void cleanup(void) {
        cleanup(data.nodes_ptr);
        cleanup(data.aabbs_ptr);
        cleanup(data.tris_ptr);
        cleanup(data.indices_ptr);
    }

    template <Location O>
    bool meta_equal(Data data, typename BVHTree<O>::Data odata) {
        return data.num_nodes == odata.num_nodes
            && data.num_faces == odata.num_faces;
    }

    template <Location O>
    void copy(typename BVHTree<O>::Data const & odata, cudaMemcpyKind src_to_dst) {
        CHECK(cudaMemcpy(data.nodes_ptr, odata.nodes_ptr,
            data.num_nodes * sizeof(Node), src_to_dst));
        CHECK(cudaMemcpy(data.aabbs_ptr, odata.aabbs_ptr,
            data.num_nodes * sizeof(AABB), src_to_dst));
        CHECK(cudaMemcpy(data.tris_ptr, odata.tris_ptr,
            data.num_faces * sizeof(Tri), src_to_dst));
        CHECK(cudaMemcpy(data.indices_ptr, odata.indices_ptr,
            data.num_faces * sizeof(uint), src_to_dst));
    }

    BVHTree() {
        data = {0, 0, nullptr, nullptr, nullptr};
    }

    BVHTree(uint num_nodes, uint num_faces) : BVHTree() {
        init(num_nodes, num_faces);
    }

public:
    template <typename IdxType, typename Vec3fType>
    static
    Ptr create(typename acc::BVHTree<IdxType, Vec3fType>::ConstPtr bvh_tree) {
        return Ptr(new BVHTree(*bvh_tree));
    }

    template<typename IdxType, typename Vec3fType>
    BVHTree(typename acc::BVHTree<IdxType, Vec3fType> const & bvh_tree);

    template <Location O>
    BVHTree& operator=(BVHTree<O> const & other) {
        typename BVHTree<O>::Data const & odata = other.cdata();
        if (!meta_equal<O>(data, odata)) {
            cleanup();
            init(odata.num_nodes, odata.num_faces);
        }

        if (L == HOST && O == HOST) copy<O>(odata, cudaMemcpyHostToHost);
        if (L == HOST && O == DEVICE) copy<O>(odata, cudaMemcpyDeviceToHost);
        if (L == DEVICE && O == HOST) copy<O>(odata, cudaMemcpyHostToDevice);
        if (L == DEVICE && O == DEVICE) copy<O>(odata, cudaMemcpyDeviceToDevice);

        return *this;
    }

    template<Location O>
    BVHTree(BVHTree<O> const & other) : BVHTree() {
        *this = other;
    }

    ~BVHTree() {
        cleanup();
    }

    Data const & cdata(void) const {
        return data;
    }

    template <typename IdxType, typename Vec3fType>
    template <class C>
    friend C acc::BVHTree<IdxType, Vec3fType>::convert(
        acc::BVHTree<IdxType, Vec3fType> const & bvh_tree);
};

CACC_NAMESPACE_END

#endif /* CACC_BVHTREE_HEADER */
