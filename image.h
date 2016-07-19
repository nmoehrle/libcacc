/*
 * Copyright (C) 2015, Nils Moehrle
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef CACC_IMAGE_HEADER
#define CACC_IMAGE_HEADER

#include <memory>

#include "defines.h"

CACC_NAMESPACE_BEGIN

template <typename T, Location L>
class Image {
public:
    typedef std::shared_ptr<Image<T, L> > Ptr;
    typedef std::shared_ptr<const Image<T, L> > ConstPtr;

    struct Data {
        int32_t width;
        int32_t height;
        uint64_t pitch;

        T * data_ptr;
    };

private:
    Data data;

    void init(int width, int height) {
        data.width = width;
        data.height = height;
        if (L == HOST) {
            data.pitch = width * sizeof(T);
            data.data_ptr = new T[width * height];
        } else {
            CHECK(cudaMallocPitch(&data.data_ptr, &data.pitch,
                width * sizeof(T), height));
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
        cleanup(data.data_ptr);
    }

    template <Location O>
    bool meta_equal(Data data, typename Image<T, O>::Data odata) {
        return data.width == odata.width
            && data.height == odata.height;
    }

public:
    Image() : data({0, 0, 0, nullptr}) {}

    Image(int width, int height) {
        init(width, height);
    }

    static Image::Ptr create() {
        return std::make_shared<Image>();
    }

    static Image::Ptr create(int width, int height) {
        return std::make_shared<Image>(width, height);
    }

    template <Location O>
    static Image::Ptr create(typename Image<T, O>::Ptr image) {
        return std::make_shared<Image>(*image);
    }

    template <Location O>
    void copy(typename Image<T, O>::Data const & odata, cudaMemcpyKind src_to_dst) {
        CHECK(cudaMemcpy2D(data.data_ptr, data.pitch, odata.data_ptr, odata.pitch,
            data.width * sizeof(T), data.height, src_to_dst));
    }

    template <Location O>
    Image& operator=(Image<T, O> const & other) {
        typename Image<T, O>::Data const & odata = other.cdata();
        if (!meta_equal<O>(data, odata)) {
            cleanup();
            init(odata.width, odata.height);
        }

        if (L == HOST && O == HOST) copy<O>(odata, cudaMemcpyHostToHost);
        if (L == HOST && O == DEVICE) copy<O>(odata, cudaMemcpyDeviceToHost);
        if (L == DEVICE && O == HOST) copy<O>(odata, cudaMemcpyHostToDevice);
        if (L == DEVICE && O == DEVICE) copy<O>(odata, cudaMemcpyDeviceToDevice);

        return *this;
    }

    template <Location O>
    Image(Image<T, O> const & other) : Image() {
        *this = other;
    }

    ~Image() {
        cleanup();
    }

    Image::Data & data_ref(void) {
        return data;
    }

    Image::Data const & cdata(void) const {
        return data;
    }
};

template <Location L>
using ByteImage = Image<uint, L>;

CACC_NAMESPACE_END

#endif /* CACC_IMAGE_HEADER */
