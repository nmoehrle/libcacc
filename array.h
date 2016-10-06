/*
 * Copyright (C) 2015, Nils Moehrle
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef CACC_ARRAY_HEADER
#define CACC_ARRAY_HEADER

#include <memory>

#include "defines.h"

CACC_NAMESPACE_BEGIN

template <typename T, Location L>
class Array {
public:
    typedef typename std::shared_ptr<Array> Ptr;

    struct Data {
        uint num_values;
        T * data_ptr;
    };

private:
    Data data;
    cudaStream_t stream;

    void init(uint num_values) {
        data.num_values = num_values;

        if (L == HOST) {
            CHECK(cudaMallocHost(&data.data_ptr, num_values * sizeof(T)));
        } else {
            CHECK(cudaMalloc(&data.data_ptr, num_values * sizeof(T)));
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
        cleanup(data.data_ptr);
    }

    template <Location O>
    bool meta_equal(Data data, typename Array<T, O>::Data odata) {
        return data.num_values == odata.num_values;
    }

    template <Location O>
    void copy(typename Array<T, O>::Data const & odata, cudaMemcpyKind src_to_dst) {
        CHECK(cudaMemcpyAsync(data.data_ptr, odata.data_ptr,
            odata.num_values * sizeof(T), src_to_dst, stream));
    }

public:
    Array() : data ({0, nullptr}), stream(cudaStreamLegacy) {};

    Array(uint num_values, cudaStream_t stream = cudaStreamLegacy)
        : data ({0, nullptr}), stream(stream) {
        init(num_values);
    }

    static Ptr create(uint num_values, cudaStream_t stream = cudaStreamLegacy) {
        return std::make_shared<Array>(num_values, stream);
    }

    template <Location O>
    static Ptr create(typename Array<T, O>::Ptr array) {
        return std::make_shared<Array>(*array);
    }

    template <Location O>
    Array& operator=(Array<T, O> const & other) {
        stream = other.current_stream();

        typename Array<T, O>::Data const & odata = other.cdata();
        if (!meta_equal<O>(data, odata)) {
            cleanup();
            init(odata.num_values);
        }

        if (L == HOST && O == HOST) copy<O>(odata, cudaMemcpyHostToHost);
        if (L == HOST && O == DEVICE) copy<O>(odata, cudaMemcpyDeviceToHost);
        if (L == DEVICE && O == HOST) copy<O>(odata, cudaMemcpyHostToDevice);
        if (L == DEVICE && O == DEVICE) copy<O>(odata, cudaMemcpyDeviceToDevice);

        if (stream == cudaStreamLegacy) sync();

        return *this;
    }

    template <Location O>
    Array(Array<T, O> const & other) : Array() {
        *this = other;
    }

    ~Array() {
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

    void null(void) {
        CHECK(cudaMemsetAsync(data.data_ptr, 0,
            data.num_values * sizeof(T), stream));

        if (stream == cudaStreamLegacy) sync();
    }
};

CACC_NAMESPACE_END

#endif /* CACC_ARRAY_HEADER */
