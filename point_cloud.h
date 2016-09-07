/*
 * Copyright (C) 2015, Nils Moehrle
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef CACC_POINTCLOUD_HEADER
#define CACC_POINTCLOUD_HEADER

#include <memory>

#include "vector.h"
#include "defines.h"

#include <cuda_runtime.h>

CACC_NAMESPACE_BEGIN

template <Location L>
class PointCloud {
public:
    typedef typename std::shared_ptr<PointCloud> Ptr;
    typedef typename std::shared_ptr<const PointCloud> ConstPtr;

    struct Data {
        uint num_vertices;
        Vec3f * vertices_ptr;
        Vec3f * normals_ptr;
        float * values_ptr;
    };

private:
    Data data;

    void init(uint num_vertices) {
        data.num_vertices = num_vertices;
        if (L == HOST) {
            data.vertices_ptr = new Vec3f[num_vertices];
            data.normals_ptr = new Vec3f[num_vertices];
            data.values_ptr = new float[num_vertices];
        } else {
            CHECK(cudaMalloc(&data.vertices_ptr, num_vertices * sizeof(Vec3f)));
            CHECK(cudaMalloc(&data.normals_ptr, num_vertices * sizeof(Vec3f)));
            CHECK(cudaMalloc(&data.values_ptr, num_vertices * sizeof(float)));
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
        cleanup(data.vertices_ptr);
        cleanup(data.normals_ptr);
        cleanup(data.values_ptr);
    }

    template <Location O>
    bool meta_equal(Data data, typename PointCloud<O>::Data odata) {
        return data.num_vertices == odata.num_vertices;
    }

    template <Location O>
    void copy(typename PointCloud<O>::Data const & odata, cudaMemcpyKind src_to_dst) {
        CHECK(cudaMemcpy(data.vertices_ptr, odata.vertices_ptr,
            data.num_vertices * sizeof(Vec3f), src_to_dst));
        CHECK(cudaMemcpy(data.normals_ptr, odata.normals_ptr,
            data.num_vertices * sizeof(Vec3f), src_to_dst));
        CHECK(cudaMemcpy(data.values_ptr, odata.values_ptr,
            data.num_vertices * sizeof(float), src_to_dst));
    }

public:
    PointCloud() : data({0, nullptr, nullptr, nullptr}) {};

    PointCloud(uint num_vertices) : PointCloud() {
        init(num_vertices);
    }

    static Ptr create(uint num_vertices) {
        return Ptr(new PointCloud(num_vertices));
    }

    template <Location O>
    static Ptr create(typename PointCloud<O>::Ptr point_cloud) {
        return Ptr(new PointCloud(*point_cloud));
    }

    template <Location O>
    PointCloud& operator=(PointCloud<O> const & other) {
        typename PointCloud<O>::Data const & odata = other.cdata();
        if (!meta_equal<O>(data, odata)) {
            cleanup();
            init(odata.num_vertices);
        }

        if (L == HOST && O == HOST) copy<O>(odata, cudaMemcpyHostToHost);
        if (L == HOST && O == DEVICE) copy<O>(odata, cudaMemcpyDeviceToHost);
        if (L == DEVICE && O == HOST) copy<O>(odata, cudaMemcpyHostToDevice);
        if (L == DEVICE && O == DEVICE) copy<O>(odata, cudaMemcpyDeviceToDevice);

        return *this;
    }

    template<Location O>
    PointCloud(PointCloud<O> const & other) : PointCloud() {
        *this = other;
    }

    ~PointCloud() {
        cleanup();
    }

    Data const & cdata(void) const {
        return data;
    }
};

CACC_NAMESPACE_END

#endif /* CACC_POINTCLOUD_HEADER */
