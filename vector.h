/*
 * Copyright (C) 2015, Nils Moehrle
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef CACC_VECTOR_HEADER
#define CACC_VECTOR_HEADER

#include "defines.h"
#include "float.h"

CACC_NAMESPACE_BEGIN

template <typename T, int N> class Vector;
template <typename T, int N> __forceinline__ __host__ __device__
float norm(Vector<T, N> const &);

template <typename T, int N>
class Vector {
protected:
    T v;
public:
    __forceinline__ __host__ __device__
    Vector() {}

    __forceinline__ __host__ __device__
    Vector(float value) {
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            v[i] = value;
        }
    }

    __forceinline__ __host__ __device__
    Vector(float x, float y) {
        v[0] = x;
        v[1] = y;
    }

    __forceinline__ __host__ __device__
    Vector(float x, float y, float z) {
        v[0] = x;
        v[1] = y;
        v[2] = z;
    }

    __forceinline__ __host__ __device__
    Vector(float x, float y, float z, float w) {
        v[0] = x;
        v[1] = y;
        v[2] = z;
        v[3] = w;
    }

    __forceinline__ __host__ __device__
    Vector(float const * values) {
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            v[i] = values[i];
        }
    }

    __forceinline__ __host__ __device__
    Vector(Vector const & other) {
        *this = other;
    }

    __forceinline__ __host__ __device__
    float & operator[] (int i) {
        return v[i];
    }

    __forceinline__ __host__ __device__
    float const & operator[] (int i) const {
        return v[i];
    }

    __forceinline__ __host__ __device__
    Vector & operator= (Vector const & other) {
        v.data = other.v.data;
        return *this;
    }

    __forceinline__ __host__ __device__
    Vector operator- (void) const {
        Vector ret;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            ret[i] = -v[i];
        }
        return ret;
    }

    __forceinline__ __host__ __device__
    Vector operator* (float const & rhs) const{
        Vector ret;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            ret[i] = v[i] * rhs;
        }
        return ret;
    }

    __forceinline__ __host__ __device__
    Vector operator/ (float const & rhs) const {
        Vector ret;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            ret[i] = v[i] / rhs;
        }
        return ret;
    }

    __forceinline__ __host__ __device__
    Vector & operator/= (float const & rhs) {
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            v[i] /= rhs;
        }
        return *this;
    }

    __forceinline__ __host__ __device__
    Vector operator+ (Vector const & rhs) const {
        Vector ret;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            ret[i] = v[i] + rhs[i];
        }
        return ret;
    }

    __forceinline__ __host__ __device__
    Vector & operator+= (Vector const & rhs) {
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            v[i] += rhs[i];
        }
        return *this;
    }

    __forceinline__ __host__ __device__
    Vector operator- (Vector const & rhs) const{
        Vector ret;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            ret[i] = v[i] - rhs[i];
        }
        return ret;
    }

    __forceinline__ __host__ __device__
    Vector normalize() {
        return *this / norm(*this);
    }
};

typedef Vector<Float4, 4> Vec4f;
typedef Vector<Float4, 3> Vec3f;
typedef Vector<Float2, 2> Vec2f;


template<typename T, int N>
__forceinline__ __host__ __device__
Vector<T, N> operator * (float lhs, Vector<T, N> const & rhs) {
    Vector<T, N> ret;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        ret[i] = lhs * rhs[i];
    }
    return ret;
}

template<typename T, int N>
__forceinline__ __host__ __device__
Vector<T, N> operator / (float lhs, Vector<T, N> const & rhs) {
    Vector<T, N> ret;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        ret[i] = lhs * rhs[i];
    }
    return ret;
}

template<typename T, int N>
__forceinline__ __host__ __device__
float dot(Vector<T, N> const & lhs, Vector<T, N> const & rhs) {
    float ret = 0.0f;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        ret += lhs[i] * rhs[i];
    }
    return ret;
}

template<typename T, int N>
__forceinline__ __host__ __device__
float square_norm(Vector<T, N> const & vec) {
    float ret = 0.0f;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        ret += vec[i] * vec[i];
    }
    return ret;
}

template<typename T, int N>
__forceinline__ __host__ __device__
float norm(Vector<T, N> const & vec) {
    return sqrt(square_norm(vec));
}

__forceinline__ __host__ __device__
Vec3f cross(Vec3f const & lhs, Vec3f const & rhs) {
    Vec3f ret;
    ret[0] = lhs[1] * rhs[2] - lhs[2] * rhs[1];
    ret[1] = lhs[2] * rhs[0] - lhs[0] * rhs[2];
    ret[2] = lhs[0] * rhs[1] - lhs[1] * rhs[0];
    return ret;
}

CACC_NAMESPACE_END

#endif /* CACC_VECTOR_HEADER */
