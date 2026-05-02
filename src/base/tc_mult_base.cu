#include "tc_mult/tc_mult.h"

#include <algorithm>
#include <cstring>

#include <cuda_runtime_api.h>

namespace
{
    constexpr int LEN_A = TC_MULT_WORDS_A;
    constexpr int LEN_B = TC_MULT_WORDS_B;
    constexpr int LEN_RESULT = TC_MULT_RESULT_WORDS;
    constexpr int HASH_THREADS_PER_BLOCK = 32;
    constexpr int MUL_THREADS_PER_BLOCK = (LEN_RESULT <= 128) ? 128 : 256;

    static int cuda_status(cudaError_t err)
    {
        return err == cudaSuccess ? TC_MULT_STATUS_SUCCESS : TC_MULT_STATUS_CUDA_ERROR;
    }

    __device__ __forceinline__ void add_u128_u64(uint64_t &hi, uint64_t &lo, uint64_t x)
    {
        uint64_t old = lo;
        lo += x;
        hi += (lo < old) ? 1ull : 0ull;
    }

    __device__ __forceinline__ uint64_t carry64_from_u128_shr32(uint64_t hi, uint64_t lo)
    {
        return (lo >> 32) | (hi << 32);
    }

    __device__ __forceinline__ void compute_coeff_naive(
        const uint32_t *__restrict__ Ash,
        const uint32_t *__restrict__ Bsh,
        int k,
        uint32_t &out_low32,
        uint64_t &out_carry64)
    {
        uint64_t hi = 0;
        uint64_t lo = 0;
        const int i0 = (k >= (LEN_B - 1)) ? (k - (LEN_B - 1)) : 0;
        const int i1 = (k < LEN_A) ? k : (LEN_A - 1);

#pragma unroll 1
        for (int i = i0; i <= i1; i++)
        {
            uint64_t prod = static_cast<uint64_t>(Ash[i]) * static_cast<uint64_t>(Bsh[k - i]);
            add_u128_u64(hi, lo, prod);
        }

        out_low32 = static_cast<uint32_t>(lo);
        out_carry64 = carry64_from_u128_shr32(hi, lo);
    }

    __global__ __launch_bounds__(MUL_THREADS_PER_BLOCK) void base_calc_kernel(
        const uint32_t *__restrict__ a,
        const uint32_t *__restrict__ b,
        uint32_t *__restrict__ result,
        size_t n)
    {
        const int lane = static_cast<int>(threadIdx.x);

        __shared__ uint32_t Bsh[LEN_B];
        __shared__ uint32_t Ash[LEN_A];
        __shared__ uint32_t L[LEN_RESULT];
        __shared__ __align__(8) uint64_t C[LEN_RESULT];

        for (int j = lane; j < LEN_B; j += static_cast<int>(blockDim.x))
            Bsh[j] = b[j];
        __syncthreads();

        for (size_t idx = static_cast<size_t>(blockIdx.x); idx < n; idx += static_cast<size_t>(gridDim.x))
        {
            for (int j = lane; j < LEN_A; j += static_cast<int>(blockDim.x))
                Ash[j] = a[idx * static_cast<size_t>(LEN_A) + static_cast<size_t>(j)];
            __syncthreads();

            for (int k = lane; k < LEN_RESULT; k += static_cast<int>(blockDim.x))
            {
                uint32_t low;
                uint64_t carry;
                compute_coeff_naive(Ash, Bsh, k, low, carry);
                L[k] = low;
                C[k] = carry;
            }
            __syncthreads();

            if (lane == 0)
            {
                uint64_t carry = 0;
#pragma unroll 1
                for (int k = 0; k < LEN_RESULT; k++)
                {
                    uint64_t s = static_cast<uint64_t>(L[k]) + (carry & 0xFFFFFFFFull);
                    L[k] = static_cast<uint32_t>(s);
                    carry = (carry >> 32) + (s >> 32) + C[k];
                }
            }
            __syncthreads();

            for (int j = lane; j < LEN_RESULT; j += static_cast<int>(blockDim.x))
                result[idx * static_cast<size_t>(LEN_RESULT) + static_cast<size_t>(j)] = L[j];
            __syncthreads();
        }
    }

    __global__ void hash_kernel(const uint32_t *result,
                                uint64_t n,
                                uint64_t *hash_block,
                                uint32_t n_per_block,
                                uint32_t start_word,
                                uint32_t word_count)
    {
        uint64_t hash = 0;
        const uint64_t base = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(n_per_block);
        const uint64_t end = min(base + static_cast<uint64_t>(n_per_block), n);

        for (uint64_t i = base; i < end; i++)
        {
            const uint32_t *row = result + i * static_cast<uint64_t>(LEN_RESULT) + start_word;
            for (uint32_t j = static_cast<uint32_t>(threadIdx.x); j < word_count; j += HASH_THREADS_PER_BLOCK)
                hash += row[j];
        }

        unsigned mask = 0xffffffffu;
        for (int offset = 16; offset > 0; offset >>= 1)
            hash += __shfl_down_sync(mask, hash, offset);
        if (threadIdx.x == 0)
            hash_block[blockIdx.x] = hash;
    }
}

extern "C" int tc_mult_base_u32(const uint32_t *d_a, const uint32_t *d_b, uint32_t *d_result, size_t n)
{
    if (d_a == nullptr || d_b == nullptr || d_result == nullptr || n == 0)
        return TC_MULT_STATUS_INVALID_ARGUMENT;

    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess)
        return cuda_status(err);

    cudaDeviceProp prop{};
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess)
        return cuda_status(err);

    int max_threads = std::max(1, prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor);
    int max_blocks = std::max(1, (max_threads + MUL_THREADS_PER_BLOCK - 1) / MUL_THREADS_PER_BLOCK);
    int blocks = std::min(max_blocks, static_cast<int>(n));

    base_calc_kernel<<<blocks, MUL_THREADS_PER_BLOCK>>>(d_a, d_b, d_result, n);
    err = cudaGetLastError();
    if (err != cudaSuccess)
        return cuda_status(err);
    return cuda_status(cudaDeviceSynchronize());
}

extern "C" uint64_t tc_mult_hash_range_u32(const uint32_t *d_result, size_t n, uint32_t start_word, uint32_t word_count)
{
    if (d_result == nullptr || n == 0 || start_word >= LEN_RESULT || word_count == 0 || start_word + word_count > LEN_RESULT)
        return 0;

    uint64_t *d_hash_block = nullptr;
    const uint32_t n_per_block = static_cast<uint32_t>(std::min<uint64_t>(512, n));
    const uint32_t blocks_cnt = static_cast<uint32_t>((n + n_per_block - 1) / n_per_block);
    if (cudaMallocManaged(&d_hash_block, blocks_cnt * sizeof(uint64_t)) != cudaSuccess)
        return 0;

    hash_kernel<<<blocks_cnt, HASH_THREADS_PER_BLOCK>>>(d_result, static_cast<uint64_t>(n), d_hash_block, n_per_block, start_word, word_count);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess)
    {
        cudaFree(d_hash_block);
        return 0;
    }

    uint64_t hash = 0;
    for (uint32_t i = 0; i < blocks_cnt; i++)
        hash += d_hash_block[i];
    cudaFree(d_hash_block);
    return hash;
}

extern "C" uint64_t tc_mult_hash_u32(const uint32_t *d_result, size_t n)
{
    return tc_mult_hash_range_u32(d_result, n, 0, LEN_RESULT);
}

extern "C" void tc_mult_fill_a_words(uint32_t *h_a, size_t n)
{
    if (h_a == nullptr)
        return;
    for (size_t i = 0; i < n; i++)
    {
        for (int j = 0; j < LEN_A; j++)
        {
            const uint32_t num = static_cast<uint32_t>(i + static_cast<size_t>(j) + 1);
            h_a[i * static_cast<size_t>(LEN_A) + static_cast<size_t>(j)] = num + (num << 8) + (num << 16) + (num << 24);
        }
    }
}

extern "C" void tc_mult_fill_b_words(uint32_t *h_b)
{
    if (h_b == nullptr)
        return;
    for (int i = 0; i < LEN_B; i++)
    {
        const uint32_t num = static_cast<uint32_t>(i * 4 + 1);
        h_b[i] = num + ((num + 1) << 8) + ((num + 2) << 16) + ((num + 3) << 24);
    }
}

extern "C" void tc_mult_fill_a_with_constant_words(uint32_t *h_a, size_t n, const uint32_t *h_src)
{
    if (h_a == nullptr || h_src == nullptr)
        return;
    for (size_t i = 0; i < n; i++)
        std::memcpy(h_a + i * static_cast<size_t>(LEN_A), h_src, static_cast<size_t>(LEN_A) * sizeof(uint32_t));
}

extern "C" void tc_mult_build_gmem_b_u8_from_b_words(uint8_t *h_gmem_b, const uint32_t *h_b)
{
    if (h_gmem_b == nullptr || h_b == nullptr)
        return;

    constexpr int K = TC_MULT_K_BYTES;
    constexpr int N = TC_MULT_N_BYTES;
    uint8_t tmp_b[K];
    static_assert(sizeof(tmp_b) == TC_MULT_WORDS_B * sizeof(uint32_t), "K must equal LEN_B*4");
    std::memcpy(tmp_b, h_b, TC_MULT_WORDS_B * sizeof(uint32_t));
    std::memset(h_gmem_b, 0, static_cast<size_t>(K) * static_cast<size_t>(N) * sizeof(uint8_t));

    for (int k = 0; k < K; k++)
    {
        for (int j = 0; j < K; j++)
        {
            const int n = j + k;
            h_gmem_b[static_cast<size_t>(n) * static_cast<size_t>(K) + static_cast<size_t>(k)] = tmp_b[j];
        }
    }
}
