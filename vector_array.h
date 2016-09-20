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

CACC_NAMESPACE_BEGIN

template <typename T, Location L>
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
    cudaStream_t stream;

    void init(uint num_cols, uint max_rows) {
        data.num_cols = num_cols;
        data.max_rows = max_rows;

        if (L == HOST) {
            CHECK(cudaMallocHost(&data.num_rows_ptr, num_cols * sizeof(uint)));
            CHECK(cudaMemset(data.num_rows_ptr, 0, num_cols * sizeof(uint)));
            CHECK(cudaMallocHost(&data.data_ptr, num_cols * max_rows * sizeof(T)));
            data.pitch = num_cols * sizeof(T);
        } else {
            CHECK(cudaMalloc(&data.num_rows_ptr, num_cols * sizeof(uint)));
            CHECK(cudaMemset(data.num_rows_ptr, 0, num_cols * sizeof(uint)));
            CHECK(cudaMallocPitch(&data.data_ptr, &data.pitch,
                num_cols * sizeof(T), max_rows));
        }
    }

    template <typename P>
    void cleanup(P * ptr) {
        if (ptr == nullptr) return;
        if (L == HOST) CHECK(cudaFreeHost(ptr));
        if (L == DEVICE) CHECK(cudaFree(ptr));
        ptr = nullptr;
    }

    void cleanup(void) {
        cleanup(data.num_rows_ptr);
        cleanup(data.data_ptr);
    }

    template <Location O>
    bool meta_equal(Data data, typename VectorArray<T, O>::Data odata) {
        return data.num_cols == odata.num_cols
            && data.max_rows == odata.max_rows;
    }

    template <Location O>
    void copy(typename VectorArray<T, O>::Data const & odata, cudaMemcpyKind src_to_dst) {
        CHECK(cudaMemcpyAsync(data.num_rows_ptr, odata.num_rows_ptr,
            odata.num_cols * sizeof(uint), src_to_dst, stream));
        CHECK(cudaMemcpy2DAsync(data.data_ptr, data.pitch, odata.data_ptr, odata.pitch,
            odata.num_cols * sizeof(T), odata.max_rows, src_to_dst, stream));
    }

public:
    VectorArray() : data ({0, 0, nullptr, nullptr, 0}), stream(cudaStreamLegacy) {};

    VectorArray(uint num_cols, uint max_rows, cudaStream_t stream = cudaStreamLegacy)
        : data ({0, 0, nullptr, nullptr, 0}), stream(stream) {
        init(num_cols, max_rows);
    }

    typedef typename std::shared_ptr<VectorArray> Ptr;

    static Ptr create(uint num_cols, uint max_rows, cudaStream_t stream = cudaStreamLegacy) {
        return std::make_shared<VectorArray>(num_cols, max_rows, stream);
    }

    template <Location O>
    static Ptr create(typename VectorArray<T, O>::Ptr vector_array) {
        return std::make_shared<VectorArray>(*vector_array);
    }

    template <Location O>
    VectorArray& operator=(VectorArray<T, O> const & other) {
        stream = other.current_stream();

        typename VectorArray<T, O>::Data const & odata = other.cdata();
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
    VectorArray(VectorArray<T, O> const & other) : VectorArray() {
        *this = other;
    }

    ~VectorArray() {
        cleanup();
    }

    Data const & cdata(void) const {
        return data;
    }

    cudaStream_t current_stream(void) const {
        return stream;
    }

    void sync(void) const {
        CHECK(cudaStreamSynchronize(stream));
    }

    void clear(void) {
        CHECK(cudaMemsetAsync(data.num_rows_ptr, 0,
            data.num_cols * sizeof(uint), stream));
    }

    void null(void) {
        CHECK(cudaMemset2DAsync(data.data_ptr, data.pitch, 0,
            data.num_cols * sizeof(T), data.max_rows, stream));
    }
};

CACC_NAMESPACE_END

#endif /* CACC_VECTORARRAY_HEADER */
