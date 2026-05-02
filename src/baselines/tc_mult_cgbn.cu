#include "tc_mult/tc_mult.h"

#include <algorithm>

#include <cuda_runtime_api.h>
#include <gmp.h>
#include <cgbn/cgbn.h>

namespace
{
    constexpr int LEN_A = TC_MULT_WORDS_A;
    constexpr int LEN_B = TC_MULT_WORDS_B;
    constexpr int LEN_RESULT = TC_MULT_RESULT_WORDS;
    constexpr int CGBN_TPI = 8;
    constexpr int BITS = LEN_A * 32;
    constexpr int THREADS_PER_BLOCK = 128;

    static_assert(LEN_A == LEN_B, "CGBN baseline requires equal operand sizes");

    using context_t = cgbn_context_t<CGBN_TPI>;
    using env_t = cgbn_env_t<context_t, BITS>;

    int cuda_status(cudaError_t err)
    {
        return err == cudaSuccess ? TC_MULT_STATUS_SUCCESS : TC_MULT_STATUS_CUDA_ERROR;
    }

    __global__ void cgbn_calc_kernel(const uint32_t *a, const uint32_t *b, uint32_t *result, size_t n)
    {
        int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / CGBN_TPI;
        if (instance >= n)
            return;

        context_t bn_context;
        env_t bn_env(bn_context);

        typename env_t::cgbn_t aa;
        typename env_t::cgbn_t bb;
        typename env_t::cgbn_wide_t rr;

        cgbn_load(bn_env, aa, reinterpret_cast<cgbn_mem_t<BITS> *>(const_cast<uint32_t *>(&a[instance * LEN_A])));
        cgbn_load(bn_env, bb, reinterpret_cast<cgbn_mem_t<BITS> *>(const_cast<uint32_t *>(&b[0])));
        cgbn_mul_wide(bn_env, rr, aa, bb);
        cgbn_store(bn_env, reinterpret_cast<cgbn_mem_t<BITS> *>(&result[instance * LEN_RESULT]), rr._low);
        cgbn_store(bn_env, reinterpret_cast<cgbn_mem_t<BITS> *>(&result[instance * LEN_RESULT + LEN_A]), rr._high);
    }
}

extern "C" int tc_mult_cgbn_u32(const uint32_t *d_a, const uint32_t *d_b, uint32_t *d_result, size_t n)
{
    if (d_a == nullptr || d_b == nullptr || d_result == nullptr || n == 0)
        return TC_MULT_STATUS_INVALID_ARGUMENT;

    int instance_per_block = THREADS_PER_BLOCK / CGBN_TPI;
    int blocks = static_cast<int>((n + static_cast<size_t>(instance_per_block) - 1) / static_cast<size_t>(instance_per_block));

    cgbn_calc_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_result, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return cuda_status(err);
    return cuda_status(cudaDeviceSynchronize());
}
