#pragma once

#include <cstdlib>
#include <iostream>

#include <cuda_runtime_api.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        std::cerr << "CUDA error in " << __FILE__ << " line "      \
                  << __LINE__ << ": " << cudaGetErrorString(err)   \
                  << " (" << err << ")" << std::endl;             \
        std::exit(EXIT_FAILURE);                                   \
    }                                                              \
} while (0)
