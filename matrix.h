/*
 * Copyright (C) 2015-2018, Nils Moehrle
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef CACC_MATRIX_HEADER
#define CACC_MATRIX_HEADER

#include "defines.h"

#include "float.h"
#include "vector.h"

CACC_NAMESPACE_BEGIN

template <typename T, int N, int M>
class Matrix {
protected:
    T v[M];
public:
    __forceinline__ __host__ __device__
    Matrix() {}

    __forceinline__ __host__ __device__
    Matrix(float value) {
        for (int i = 0; i < M; ++i) {
            #pragma unroll
            for (int j = 0; j < N; ++j) {
                v[i][j] = value;
            }
        }
    }
    __forceinline__ __host__ __device__
    Matrix(float const * values) {
        for (int i = 0; i < M; ++i) {
            #pragma unroll
            for (int j = 0; j < N; ++j) {
                v[i][j] = values[i * M + j];
            }
        }
    }

    __forceinline__ __host__ __device__
    Matrix(Matrix const & other) {
        *this = other;
    }

    __forceinline__ __host__ __device__
    T & operator[] (int i) {
        return v[i];
    }

    __forceinline__ __host__ __device__
    T const & operator[] (int i) const {
        return v[i];
    }

    __forceinline__ __host__ __device__
    Matrix & operator= (Matrix const & other) {
        for (int i = 0; i < M; ++i) {
            v[i].data = other.v[i].data;
        }
        return *this;
    }

    __forceinline__ __device__
    Vector<T, N> operator* (Vector<T, N> const & rhs) const {
        Vector<T, N> ret;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            float sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < M; ++j) {
                sum += v[i][j] * rhs[j];
            }
            ret[i] = sum;
        }
        return ret;
    }

    __forceinline__ __host__ __device__
    Matrix operator/ (float const & rhs) {
        Matrix ret;
        for (int i = 0; i < M; ++i) {
            #pragma unroll
            for (int j = 0; j < N; ++j) {
                ret[i][j] = v[i][j] / rhs;
            }
        }
        return ret;
    }

    __forceinline__ __host__ __device__
    Matrix & operator/= (float const & rhs) {
        for (int i = 0; i < M; ++i) {
            #pragma unroll
            for (int j = 0; j < N; ++j) {
                v[i][j] /= rhs;
            }
        }
        return *this;
    }
};

typedef Matrix<Float2, 2, 2> Mat2f;
typedef Matrix<Float4, 3, 3> Mat3f;
typedef Matrix<Float4, 4, 4> Mat4f;

template<typename T, int N, int M>
__forceinline__ __device__
Vector<T, N - 1> mult (Matrix<T, N, M> const & m, Vector<T, N - 1> const & v, float w) {
    Vector<T, N - 1> ret;
    #pragma unroll
    for (int i = 0; i < N - 1; ++i) {
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < M - 1; ++j) {
            sum += m[i][j] * v[j];
        }
        ret[i] = sum + m[i][M - 1] * w;
    }
    return ret;
}

template<typename T, int N>
__forceinline__ __device__
Vector<T, N> mult (Matrix<T, N, N> const & m, Vector<T, N> const & v) {
    Vector<T, N> ret;
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < N; ++j) {
            sum += m[i][j] * v[j];
        }
        ret[i] = sum;
    }
    return ret;
}

template<typename T, int N>
__forceinline__ __device__
float det(Matrix<T, N, N> const & m);

template<typename T, int N>
__forceinline__ __device__
float trace(Matrix<T, N, N> const & m);

template<>
__forceinline__ __device__
float det(Matrix<Float2, 2, 2> const & m) {
    return m[0][0] * m[1][1] - m[0][1] * m[1][0];
}

template<>
__forceinline__ __device__
float trace(Matrix<Float2, 2, 2> const & m) {
    return m[0][0] + m[1][1];
}

CACC_NAMESPACE_END

#endif /* CACC_VECTOR_HEADER */
