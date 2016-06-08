/*
 * Copyright (C) 2015, Nils Moehrle
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef CACC_VECTORARRAY_HEADER
#define CACC_VECTORARRAY_HEADER

#include <memory>

#include "defines.h"

#include <cuda_runtime.h>

CACC_NAMESPACE_BEGIN

template <Location L, typename T>
class VectorArray {
public:
    struct Data {
        uint num_cols;
        uint max_rows;
        uint * num_rows_ptr;
        T * data_ptr;
        size_t pitch;
    };

private:
    Data data;

    void init(uint num_cols, uint max_rows) {
        data.num_cols = num_cols;
        data.max_rows = max_rows;

        if (L == HOST) {
            data.num_rows_ptr = new uint[num_cols]();
            data.data_ptr = new T[num_cols * max_rows];
            data.pitch = num_cols * sizeof(T);
        } else {
            CHECK(cudaMalloc(&data.num_rows_ptr, num_cols * sizeof(uint)));
            CHECK(cudaMemset(data.num_rows_ptr, 0, num_cols * sizeof(uint)));
            CHECK(cudaMallocPitch(&data.data_ptr, &data.pitch, num_cols * sizeof(T), max_rows));
        }
    }

    template <typename P>
    void cleanup(P * ptr) {
        if (ptr == nullptr) return;
        if (L == HOST) delete[] ptr;
        if (L == DEVICE) CHECK(cudaFree(ptr));
    }

    void cleanup(void) {
        cleanup(data.num_rows_ptr);
        cleanup(data.data_ptr);
    }

    template <Location O>
    bool meta_equal(Data data, typename VectorArray<O, T>::Data odata) {
        return data.num_cols == odata.num_cols
            && data.max_rows == odata.max_rows;
    }

public:
    VectorArray() {
        data = {0, 0, nullptr, nullptr, 0};
    }

    VectorArray(uint num_cols, uint max_rows) : VectorArray() {
        init(num_cols, max_rows);
    }

    typedef typename std::shared_ptr<VectorArray> Ptr;

    static Ptr create(uint num_cols, uint max_rows) {
        return Ptr(new VectorArray(num_cols, max_rows));
    }

    template <Location O>
    static Ptr create(typename VectorArray<O, T>::Ptr vector_array) {
        return Ptr(new VectorArray(*vector_array));
    }

    template <Location O>
    void copy(typename VectorArray<O, T>::Data const & odata, cudaMemcpyKind src_to_dst) {
        CHECK(cudaMemcpy(data.num_rows_ptr, odata.num_rows_ptr,
            data.num_cols * sizeof(uint), src_to_dst));
        CHECK(cudaMemcpy2D(data.data_ptr, data.pitch, odata.data_ptr, odata.pitch,
            odata.num_cols * sizeof(T), odata.max_rows, src_to_dst));
    }

    template <Location O>
    VectorArray& operator=(VectorArray<O, T> const & other) {
        typename VectorArray<O, T>::Data const & odata = other.cdata();
        if (!meta_equal<O>(data, odata)) {
            cleanup();
            init(odata.num_cols, odata.max_rows);
        }

        if (L == HOST && O == HOST) copy<O>(odata, cudaMemcpyHostToHost);
        if (L == HOST && O == DEVICE) copy<O>(odata, cudaMemcpyDeviceToHost);
        if (L == DEVICE && O == HOST) copy<O>(odata, cudaMemcpyHostToDevice);
        if (L == DEVICE && O == DEVICE) copy<O>(odata, cudaMemcpyDeviceToDevice);

        return *this;
    }

    template<Location O>
    VectorArray(VectorArray<O, T> const & other) : VectorArray() {
        *this = other;
    }

    ~VectorArray() {
        cleanup();
    }

    Data const & cdata(void) const {
        return data;
    }
};

CACC_NAMESPACE_END

#endif /* CACC_VECTORARRAY_HEADER */
