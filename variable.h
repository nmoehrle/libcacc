/*
 * Copyright (C) 2015-2018, Nils Moehrle
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef CACC_VARIABLE_HEADER
#define CACC_VARIABLE_HEADER

#include "defines.h"

CACC_NAMESPACE_BEGIN

template <typename T, Location L>
class Variable {
private:
    T * ptr;

    void init() {
        if (L == HOST) {
            CHECK(cudaMallocHost(&ptr, sizeof(T)));
        } else {
            CHECK(cudaMalloc(&ptr, sizeof(T)));
        }
    }

    template <typename P>
    void cleanup(P * ptr) {
        if (ptr == nullptr) return;
        if (L == HOST) CHECK(cudaFreeHost(ptr));
        if (L == DEVICE) CHECK(cudaFree(ptr));
        ptr = nullptr;
    }

    void copy(const T * optr, cudaMemcpyKind src_to_dst) {
        CHECK(cudaMemcpy(ptr, optr, sizeof(T), src_to_dst));
    }

    Variable() { init(); }

public:
    Variable(T const & v) : Variable() {
        if (L == HOST) {
            copy(&v, cudaMemcpyHostToHost);
        } else {
            copy(&v, cudaMemcpyHostToDevice);
        }
    }

    template <Location O>
    Variable& operator=(Variable<T, O> const & other) {
        if (L == HOST && O == HOST) copy(other.cptr(), cudaMemcpyHostToHost);
        if (L == HOST && O == DEVICE) copy(other.cptr(), cudaMemcpyDeviceToHost);
        if (L == DEVICE && O == HOST) copy(other.cptr(), cudaMemcpyHostToDevice);
        if (L == DEVICE && O == DEVICE) copy(other.cptr(), cudaMemcpyDeviceToDevice);

        return *this;
    }

    template <Location O>
    Variable(Variable<T, O> const & other) : Variable() {
        *this = other;
    }

    ~Variable() {
        cleanup(ptr);
    }

    T * cptr(void) const {
        static_assert(L == DEVICE, "Only available for device variables");
        return ptr;
    }

    T & ref(void) {
        static_assert(L == HOST, "Only available for host variables");
        return *ptr;
    }
};

CACC_NAMESPACE_END

#endif /* CACC_VARIABLE_HEADER */
