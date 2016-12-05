/*
 * Copyright (C) 2015, Nils Moehrle
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef CACC_BUFFERTEXTURE_HEADER
#define CACC_BUFFERTEXTURE_HEADER

#include "defines.h"

#include <cuda_runtime_api.h>

CACC_NAMESPACE_BEGIN

template<typename T>
class BufferTexture {
public:
    class Accessor {
    private:
        cudaTextureObject_t tex;
    public:
        __device__ __forceinline__
        T operator[](int idx) {
            return tex1Dfetch<T>(tex, idx);
        }
    };

private:
    cudaTextureObject_t tex;
protected:
    void initialize(T *ptr, size_t size);
    BufferTexture() {}
public:
    BufferTexture(T *ptr, size_t size);
    ~BufferTexture();
    Accessor accessor() { return Accessor(tex); };
};

template<typename T>
class MappedBufferTexture : public BufferTexture<T> {
private:
    cudaGraphicsResource_t &res;
    cudaStream_t stream;
public:
    MappedBufferTexture(GraphicsResource *resource,
        cudaStream_t = cudaStreamLegacy);
    ~MappedBufferTexture();
};

template<typename T>
BufferTexture<T>::BufferTexture(T *ptr, size_t size) {
    initialize(ptr, size);
}

template<typename T>
MappedBufferTexture<T>::MappedBufferTexture(GraphicsResource *resource,
    cudaStream_t steam) : res(resource->ptr()), stream(stream)
{
    CHECK(cudaGraphicsMapResources(1, &res, stream));
    T *ptr;
    size_t size;
    CHECK(cudaGraphicsResourceGetMappedPointer((void **)&ptr, &size, res));
    this->initialize(ptr, size);
}

template<typename T>
BufferTexture<T>::~BufferTexture() {
    CHECK(cudaDestroyTextureObject(tex));
}

template<typename T>
MappedBufferTexture<T>::~MappedBufferTexture() {
    CHECK(cudaGraphicsUnmapResources(1, &res, stream));
}

template<typename T>
void
BufferTexture<T>::initialize(T *ptr, size_t size) {
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = ptr;
    resDesc.res.linear.desc = cudaCreateChannelDesc<T>();
    resDesc.res.linear.sizeInBytes = sizeof(T) * size;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    CHECK(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));
}

CACC_NAMESPACE_END

#endif /* CACC_BUFFERTEXTURE_HEADER */
