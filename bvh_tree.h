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

#include "defines.h"

#include "vector.h"
#include "primitives.h"
#include "buffer_texture.h"

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

    struct Accessor {
        BufferTexture<uint4>::Accessor nodes;
        BufferTexture<float4>::Accessor aabbs;
        BufferTexture<float4>::Accessor tris;
        uint *indices_ptr;

        __device__ __forceinline__
        Node load_node(uint idx) const {
            BVHTree<DEVICE>::Node node;
            node.rllf = nodes[idx];
            return node;
        }

        __device__ __forceinline__
        AABB load_aabb(uint idx) const {
            AABB aabb;
            float4 min = aabbs[2 * idx + 0];
            aabb.min = Vec3f(min.x, min.y, min.z);
            float4 max = aabbs[2 * idx + 1];
            aabb.max = Vec3f(max.x, max.y, max.z);
            return aabb;
        }

        __device__ __forceinline__
        Tri load_tri(uint idx) const {
            Tri tri;
            float4 a = tris[3 * idx + 0];
            tri.a = Vec3f(a.x, a.y, a.z);
            float4 b = tris[3 * idx + 1];
            tri.b = Vec3f(b.x, b.y, b.z);
            float4 c = tris[3 * idx + 2];
            tri.c = Vec3f(c.x, c.y, c.z);
            return tri;
        }
    };

private:
    struct Data {
        uint num_nodes;
        uint num_faces;
        uint *indices_ptr;
        Tri *tris_ptr;
        AABB *aabbs_ptr;
        Node *nodes_ptr;
    };

    Data data;

    struct Textures {
        BufferTexture<uint4> *nodes;
        BufferTexture<float4> *aabbs;
        BufferTexture<float4> *tris;
    };

    Textures textures;

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

        if (L == DEVICE) initialize_textures();
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

        if (L == DEVICE) cleanup_textures();
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

    void initialize_textures(void) {
        static_assert(sizeof(BVHTree<DEVICE>::Node) == sizeof(uint4), "");
        static_assert(sizeof(AABB) == 2 * sizeof(float4), "");
        static_assert(sizeof(Tri) == 3 * sizeof(float4), "");
        textures.nodes = new BufferTexture<uint4>(
            (uint4 *) data.nodes_ptr, data.num_nodes);
        textures.aabbs = new BufferTexture<float4>(
            (float4 *) data.aabbs_ptr, 2 * data.num_nodes);
        textures.tris = new BufferTexture<float4>(
            (float4 *) data.tris_ptr, 3 * data.num_faces);
    }

    template <typename T>
    void cleanup(BufferTexture<T> * ptr) {
        if (ptr == nullptr) return;
        delete ptr;
        ptr = nullptr;
    }

    void cleanup_textures(void) {
        cleanup(textures.nodes);
        cleanup(textures.aabbs);
        cleanup(textures.tris);
    }

    BVHTree() {
        data = {0, 0, nullptr, nullptr, nullptr, nullptr};
        textures = {nullptr, nullptr, nullptr};
    }

    BVHTree(uint num_nodes, uint num_faces) : BVHTree() {
        init(num_nodes, num_faces);
    }

    template <Location O> friend class BVHTree;

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
        typename BVHTree<O>::Data const & odata = other.data;
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

    Accessor accessor(void) const;

    template <typename IdxType, typename Vec3fType>
    template <class C>
    friend C acc::BVHTree<IdxType, Vec3fType>::convert(
        acc::BVHTree<IdxType, Vec3fType> const & bvh_tree);
};

template<> inline
BVHTree<DEVICE>::Accessor
BVHTree<DEVICE>::accessor(void) const {
    return {
        textures.nodes->accessor(),
        textures.aabbs->accessor(),
        textures.tris->accessor(),
        data.indices_ptr
    };
}

CACC_NAMESPACE_END

#endif /* CACC_BVHTREE_HEADER */
