#pragma once

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime_api.h>

#include "tc_mult/tc_mult.h"

#define TC_MULT_EXAMPLE_CUDA_CHECK(call)                                                     \
    do                                                                                        \
    {                                                                                         \
        cudaError_t err__ = (call);                                                           \
        if (err__ != cudaSuccess)                                                             \
        {                                                                                     \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
            std::exit(EXIT_FAILURE);                                                          \
        }                                                                                     \
    } while (0)

static inline size_t parse_n(int argc, char **argv)
{
    size_t n = TC_MULT_BLOCK_M;
    if (argc > 1)
        n = static_cast<size_t>(std::strtoull(argv[1], nullptr, 10));
    if (n == 0 || n % TC_MULT_BLOCK_M != 0)
    {
        std::fprintf(stderr, "n must be a positive multiple of %d\n", TC_MULT_BLOCK_M);
        std::exit(EXIT_FAILURE);
    }
    return n;
}

struct device_inputs
{
    explicit device_inputs(size_t n_) : n(n_), h_a(n * TC_MULT_WORDS_A), h_b(TC_MULT_WORDS_B)
    {
        tc_mult_fill_a_words(h_a.data(), n);
        tc_mult_fill_b_words(h_b.data());
        TC_MULT_EXAMPLE_CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(uint32_t)));
        TC_MULT_EXAMPLE_CUDA_CHECK(cudaMalloc(&d_b, h_b.size() * sizeof(uint32_t)));
        TC_MULT_EXAMPLE_CUDA_CHECK(cudaMalloc(&d_result, n * TC_MULT_RESULT_WORDS * sizeof(uint32_t)));
        TC_MULT_EXAMPLE_CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        TC_MULT_EXAMPLE_CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    ~device_inputs()
    {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_result);
        cudaFree(d_gmem_b);
    }

    void ensure_gmem_b()
    {
        if (d_gmem_b != nullptr)
            return;
        std::vector<uint8_t> h_gmem_b(TC_MULT_GMEM_B_BYTES);
        tc_mult_build_gmem_b_u8_from_b_words(h_gmem_b.data(), h_b.data());
        TC_MULT_EXAMPLE_CUDA_CHECK(cudaMalloc(&d_gmem_b, h_gmem_b.size() * sizeof(uint8_t)));
        TC_MULT_EXAMPLE_CUDA_CHECK(cudaMemcpy(d_gmem_b, h_gmem_b.data(), h_gmem_b.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));
    }

    void clear_result()
    {
        TC_MULT_EXAMPLE_CUDA_CHECK(cudaMemset(d_result, 0, n * TC_MULT_RESULT_WORDS * sizeof(uint32_t)));
    }

    size_t n;
    std::vector<uint32_t> h_a;
    std::vector<uint32_t> h_b;
    uint32_t *d_a = nullptr;
    uint32_t *d_b = nullptr;
    uint32_t *d_result = nullptr;
    uint8_t *d_gmem_b = nullptr;
};

static inline void require_success(const char *name, int status)
{
    if (status != TC_MULT_STATUS_SUCCESS)
    {
        std::fprintf(stderr, "%s failed with status %d\n", name, status);
        std::exit(EXIT_FAILURE);
    }
}
