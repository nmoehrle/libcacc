/*
 * Copyright (C) 2015, Nils Moehrle
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef CACC_ARRAYTEXTURE_HEADER
#define CACC_ARRAYTEXTURE_HEADER

#include "defines.h"

#include "graphics_resource.h"

#include <cuda_runtime_api.h>

CACC_NAMESPACE_BEGIN

template<typename T>
class ArrayTexture {
public:
    class YAccessor {
    private:
        cudaTextureObject_t tex;
        int x;
    public:
        __device__ __forceinline__
        YAccessor(cudaTextureObject_t tex, int x) : tex(tex), x(x) {}
        __device__ __forceinline__
        T operator[](int y) {
            return tex2D<T>(tex, x, y);
        }
    };

    class Accessor {
    private:
        cudaTextureObject_t tex;
    public:
        Accessor(cudaTextureObject_t tex) : tex(tex) {}
        __device__ __forceinline__
        YAccessor operator[](int x) {
            return YAccessor(tex, x);
        }
    };

private:
    cudaTextureObject_t tex;
protected:
    void initialize(cudaArray_t array);
    ArrayTexture() {}
public:
    ArrayTexture(cudaArray_t array);
    ~ArrayTexture();
    Accessor accessor() { return Accessor(tex); };
};

template<typename T>
class MappedArrayTexture : public ArrayTexture<T> {
private:
    cudaGraphicsResource_t &res;
    cudaStream_t stream;
public:
    MappedArrayTexture(GraphicsResource *resource,
        cudaStream_t = cudaStreamLegacy);
    ~MappedArrayTexture();
};

template<typename T>
MappedArrayTexture<T>::MappedArrayTexture(GraphicsResource *resource,
   cudaStream_t stream) : res(resource->ptr()), stream(stream)
{
    CHECK(cudaGraphicsMapResources(1, &res, stream));
    cudaArray_t array;
    CHECK(cudaGraphicsSubResourceGetMappedArray(&array, res, 0, 0));
    this->initialize(array);
}

template<typename T>
ArrayTexture<T>::ArrayTexture(cudaArray *array) {
    initialize(array);
}

template<typename T>
ArrayTexture<T>::~ArrayTexture() {
    CHECK(cudaDestroyTextureObject(tex));
}

template<typename T>
MappedArrayTexture<T>::~MappedArrayTexture() {
    CHECK(cudaGraphicsUnmapResources(1, &res, stream));
}

template<typename T>
void
ArrayTexture<T>::initialize(cudaArray *array) {
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    CHECK(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));
}

CACC_NAMESPACE_END

#endif /* CACC_ARRAYTEXTURE_HEADER */
