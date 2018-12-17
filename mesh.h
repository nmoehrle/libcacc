/*
 * Copyright (C) 2015-2018, Nils Moehrle
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef CACC_MESH_HEADER
#define CACC_MESH_HEADER

#include <memory>

#include "defines.h"
#include "vector.h"

CACC_NAMESPACE_BEGIN

/* Mesh optimized for consecutive face traversal. */
template <int N, Location L>
class Mesh {

public:
    typedef std::shared_ptr<Mesh<N, L> > Ptr;
    typedef std::shared_ptr<const Mesh<N, L> > ConstPtr;

    struct Data {
        uint num_faces;
        uint * vid_face_ptr[N];

        uint num_verts;
        Vec3f * verts_ptr;
    };

private:
    Data data;

    void init(uint num_faces, uint num_verts) {
        data.num_faces = num_faces;
        data.num_verts = num_verts;
        if (L == HOST) {
            for (int i = 0; i < N; ++i) {
                data.vid_face_ptr[i] = new uint[num_faces];
            }
            data.verts_ptr = new Vec3f[num_verts];
        } else {
            for (int i = 0; i < N; ++i) {
                CHECK(cudaMalloc(&data.vid_face_ptr[i], num_faces * sizeof(uint)));
            }
            CHECK(cudaMalloc(&data.verts_ptr, num_verts * sizeof(Vec3f)));
        }
    }

    template <typename P>
    void cleanup(P * ptr) {
        if (ptr == nullptr) return;
        if (L == HOST) delete[] ptr;
        if (L == DEVICE) CHECK(cudaFree(ptr));
        ptr = nullptr;
    }

    void cleanup(void) {
        for (int i = 0; i < N; ++i) {
            cleanup(data.vid_face_ptr[i]);
        }
        cleanup(data.verts_ptr);
    }

    template <Location O>
    bool meta_equal(Data data, typename Mesh<O, L>::Data odata) {
        return data.num_faces == odata.num_faces
            && data.num_verts == odata.num_verts;
    }

    template <Location O>
    void copy(typename Mesh<O, L>::Data const & odata, cudaMemcpyKind src_to_dst) {
        for (int i = 0; i < N; ++i) {
            CHECK(cudaMemcpy(data.vid_face_ptr[i], odata.vid_face_ptr[i],
                data.num_faces * sizeof(uint), src_to_dst));
        }
        CHECK(cudaMemcpy(data.verts_ptr, odata.verts_ptr,
            data.num_verts * sizeof(Vec3f), src_to_dst));
    }

public:
    Mesh() {
        data.num_faces = 0;
        data.num_verts = 0;
        data.verts_ptr = nullptr;
        for (int i = 0; i < N; ++i) {
            data.vid_face_ptr[i] = nullptr;
        }
    }

    static Mesh::Ptr create() {
        return std::make_shared<Mesh>();
    }

    template<typename O>
    static Mesh::Ptr create(typename Mesh<N, O>::Ptr mesh) {
        return std::make_shared<Mesh>(mesh);
    }

    template <Location O>
    Mesh & operator=(Mesh<N, O> const & other) {
        typename Mesh<N, O>::Data const & odata = other.cdata();
        if (meta_equal<O>(data, odata)) {
            cleanup();
            init(odata.num_faces, odata.num_verts);
        }

        if (L == HOST && O == HOST) copy<O>(odata, cudaMemcpyHostToHost);
        if (L == HOST && O == DEVICE) copy<O>(odata, cudaMemcpyDeviceToHost);
        if (L == DEVICE && O == HOST) copy<O>(odata, cudaMemcpyHostToDevice);
        if (L == DEVICE && O == DEVICE) copy<O>(odata, cudaMemcpyDeviceToDevice);

        return *this;
    }

    template<Location O>
    Mesh(Mesh<N, O> const & other) : Mesh() {
        *this = other;
    }

    ~Mesh() {
        cleanup();
    }

    Mesh<N, L>::Data const & cdata(void) const {
        return data;
    }
};

template <Location L>
using TriangleMesh = Mesh<3, L>;

CACC_NAMESPACE_END

#endif /* CACC_MESH_HEADER */
