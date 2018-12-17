/*
 * Copyright (C) 2015-2018, Nils Moehrle
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef CACC_KDTREE_HEADER
#define CACC_KDTREE_HEADER

#include "acc/kd_tree.h"

#include "defines.h"

#include "vector.h"
#include "buffer_texture.h"

CACC_NAMESPACE_BEGIN

template <uint K, Location L>
class KDTree {
public:
    typedef typename std::conditional<
        K <= 2,
        typename cacc::Vector<Float2, K>,
        typename cacc::Vector<Float4, K>
    >::type Vertex;

    typedef std::shared_ptr<KDTree> Ptr;

    const uint NAI = std::numeric_limits<uint>::max();

    #pragma __align__(64)
    struct Node {
        union {
            struct {
                uint vid;
                uint dim;
                uint left;
                uint right;
            };
            uint4 rldv;
        };
    };

    struct Accessor {
        BufferTexture<uint4>::Accessor nodes;
        uint num_verts;
        Vertex * verts_ptr;

        __device__ __forceinline__
        Node load_node(uint idx) const {
            Node node;
            node.rldv = nodes[idx];
            return node;
        }

        __device__ __forceinline__
        Vertex load_vertex(uint idx) const {
            return verts_ptr[idx];
        }
    };

    template <uint K2, Location O> friend class KDTree;

private:
    struct Data {
        uint num_nodes;
        uint num_verts;
        Node * nodes_ptr;
        Vertex * verts_ptr;
    };

    Data data;

    struct Textures {
        BufferTexture<uint4> * nodes;
    };

    Textures textures;

    void init(uint num_nodes, uint num_verts) {
        data.num_nodes = num_nodes;
        data.num_verts = num_verts;
        if (L == HOST) {
            data.nodes_ptr = new Node[num_nodes];
            data.verts_ptr = new Vertex[num_verts];
        } else {
            CHECK(cudaMalloc(&data.nodes_ptr, num_nodes * sizeof(Node)));
            CHECK(cudaMalloc(&data.verts_ptr, num_verts * sizeof(Vertex)));
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
        cleanup(data.verts_ptr);
    }

    template <Location O>
    bool meta_equal(Data data, typename KDTree<K, O>::Data odata) {
        return data.num_nodes == odata.num_nodes
            && data.num_verts == odata.num_verts;
    }

    template <Location O>
    void copy(typename KDTree<K, O>::Data const & odata, cudaMemcpyKind src_to_dst) {
        CHECK(cudaMemcpy(data.nodes_ptr, odata.nodes_ptr,
            data.num_nodes * sizeof(Node), src_to_dst));
        CHECK(cudaMemcpy(data.verts_ptr, odata.verts_ptr,
            data.num_verts * sizeof(Vertex), src_to_dst));
    }

    void initialize_textures(void) {
        static_assert(sizeof(Node) == sizeof(uint4), "");
        textures.nodes = new BufferTexture<uint4>(
            (uint4 *) data.nodes_ptr, data.num_nodes);
    }

    template <typename T>
    void cleanup(BufferTexture<T> * ptr) {
        if (ptr == nullptr) return;
        delete ptr;
        ptr = nullptr;
    }

    void cleanup_textures(void) {
        cleanup(textures.nodes);
    }

    KDTree() {
        data = {0, 0, nullptr, nullptr};
    }

    KDTree(uint num_nodes, uint num_verts) : KDTree() {
        init(num_nodes, num_verts);
    }

public:
    template <typename IdxType>
    static
    Ptr create(typename acc::KDTree<K, IdxType>::ConstPtr kd_tree) {
        return Ptr(new KDTree(*kd_tree));
    }

    template <typename IdxType>
    KDTree(typename acc::KDTree<K, IdxType> const & kd_tree);

    template <Location O>
    KDTree& operator=(KDTree<K, O> const & other) {
        typename KDTree<K, O>::Data const & odata = other.cdata();
        if (!meta_equal<O>(data, odata)) {
            cleanup();
            init(odata.num_nodes, odata.num_verts);
        }

        if (L == HOST && O == HOST) copy<O>(odata, cudaMemcpyHostToHost);
        if (L == HOST && O == DEVICE) copy<O>(odata, cudaMemcpyDeviceToHost);
        if (L == DEVICE && O == HOST) copy<O>(odata, cudaMemcpyHostToDevice);
        if (L == DEVICE && O == DEVICE) copy<O>(odata, cudaMemcpyDeviceToDevice);

        return *this;
    }

    template<Location O>
    KDTree(KDTree<K, O> const & other) : KDTree() {
        *this = other;
    }

    ~KDTree() {
        cleanup();
    }

    Data const & cdata(void) const {
        return data;
    }

    Accessor accessor(void) const;

    template <unsigned K2, typename IdxType>
    template <class C>
    friend C acc::KDTree<K2, IdxType>::convert(
        acc::KDTree<K2, IdxType> const & kd_tree);
};

template<> inline
KDTree<3u, DEVICE>::Accessor
KDTree<3u, DEVICE>::accessor(void) const {
    return {
        textures.nodes->accessor(),
        data.num_verts,
        data.verts_ptr
    };
}

CACC_NAMESPACE_END

#endif /* CACC_KDTREE_HEADER */
