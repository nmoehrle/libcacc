/*
 * Copyright (C) 2015, Nils Moehrle
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef CACC_UTIL_HEADER
#define CACC_UTIL_HEADER

#include <chrono>
#include <thread>
#include <iostream>

#include "defines.h"

#define TERM_WIDTH 76

CACC_NAMESPACE_BEGIN

void
print_cuda_devices(void) {
    int count;
    CHECK(cudaGetDeviceCount(&count));
    std::cout << "Found " << count << " CUDA Devices:" << std::endl;
    std::cout << std::string(TERM_WIDTH, '=') << std::endl;
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop;
        CHECK(cudaGetDeviceProperties(&prop, i));
        std::cout << std::string(TERM_WIDTH, '-') << std::endl;
        std::cout << "Device:\t" << prop.name << std::endl;
        std::cout << std::string(TERM_WIDTH, '-') << std::endl;
        std::cout << "\tCompute Capability:\t" << prop.major <<
            "." << prop.minor << std::endl
            << "\tMultiprocessor Count:\t"
            << prop.multiProcessorCount << std::endl
            << "\tGPU Clock Rate:\t\t"
            << prop.clockRate / 1000 << " Mhz" << std::endl
            << "\tTotal Global Memory:\t"
            << prop.totalGlobalMem / (2 << 20) << " MB" << std::endl
            << "\tL2 Cache Size:\t\t"
            << prop.l2CacheSize / (2 << 10) << " KB" << std::endl
            << "\tMax Block Size:\t\t"
            << prop.maxThreadsDim[0] << "x"
            << prop.maxThreadsDim[1] << "x"
            << prop.maxThreadsDim[2] << std::endl
            << "\tMax Threads Per Block:\t"
            << prop.maxThreadsPerBlock << std::endl
            << "\tShared Memory Size:\t"
            << prop.sharedMemPerBlock / (2 << 10) << " KB" << std::endl;
    }
    std::cout << std::string(TERM_WIDTH, '=') << std::endl;
}

int
get_cuda_device(int major, int minor) {
    int device;
    cudaDeviceProp prop;
    prop.major = major;
    prop.minor = minor;
    CHECK(cudaChooseDevice(&device, &prop));
    return device;
}

int
get_cuda_devices(int major, int minor, std::vector<int> *devices) {
    int count;
    int num_devices = 0;
    CHECK(cudaGetDeviceCount(&count));
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop;
        CHECK(cudaGetDeviceProperties(&prop, i));
        if (prop.major > major || prop.major == major && prop.minor >= minor) {
            devices->push_back(i);
            num_devices += 1;
        }
    }
    return num_devices;
}

int
select_cuda_device(int major, int minor) {
    int device = get_cuda_device(major, minor);
    CHECK(cudaSetDevice(device));
    return device;
}

std::string
device_name(int device) {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, device));
    return prop.name;
}

void
set_cuda_device(int device) {
    CHECK(cudaSetDevice(device));
}

void
set_cuda_gl_device(int device) {
    CHECK(cudaSetDevice(device));
}

template <class Rep, class Period>
void
sync(cudaStream_t stream, cudaEvent_t event,
    std::chrono::duration<Rep, Period> sleep = std::chrono::milliseconds(1))
{
    CHECK(cudaEventRecord(event, stream));
    while (cudaEventQuery(event) != cudaSuccess) {
        std::this_thread::sleep_for(sleep);
    }
}

CACC_NAMESPACE_END

#endif /* CACC_UTIL_HEADER */
