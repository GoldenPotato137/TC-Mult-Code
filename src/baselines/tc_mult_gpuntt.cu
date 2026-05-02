#include "tc_mult/tc_mult.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include <cuda_runtime_api.h>

#include <gpuntt/ntt_merge/ntt.cuh>
#include <gpuntt/common/nttparameters.cuh>

namespace
{
    constexpr int LEN_A = TC_MULT_WORDS_A;
    constexpr int LEN_B = TC_MULT_WORDS_B;
    constexpr int LEN_RESULT = TC_MULT_RESULT_WORDS;

    static_assert(LEN_A == LEN_B, "GPU-NTT baseline requires equal operand sizes");

    constexpr int ceil_log2_int(int x)
    {
        int p = 0;
        int v = 1;
        while (v < x)
        {
            v <<= 1;
            p++;
        }
        return p;
    }

    static constexpr int DIGITS16 = LEN_A * 2;
    static constexpr int REQUIRED_DIGITS16 = DIGITS16 * 2;
    static constexpr int N_POWER = ceil_log2_int(REQUIRED_DIGITS16);
    static constexpr int N = 1 << N_POWER;
    static constexpr int THREADS_PACK = 256;
    static constexpr int MAX_BATCH = 8192;

    static_assert(N >= REQUIRED_DIGITS16, "NTT size must cover linear convolution");
    static_assert(N_POWER > 0 && N_POWER < 29, "GPU-NTT merge NTT requires 1 <= n_power <= 28");

    int cuda_status(cudaError_t err)
    {
        return err == cudaSuccess ? TC_MULT_STATUS_SUCCESS : TC_MULT_STATUS_CUDA_ERROR;
    }

    __global__ void pack_a_u32_to_u64_poly16_kernel(const uint32_t *__restrict__ a,
                                                    Data64 *__restrict__ poly,
                                                    int batch_size)
    {
        const int inst = static_cast<int>(blockIdx.x);
        if (inst >= batch_size)
            return;

        const int tid = static_cast<int>(threadIdx.x);
        if (tid < LEN_A)
        {
            const uint32_t limb = a[static_cast<size_t>(inst) * static_cast<size_t>(LEN_A) + static_cast<size_t>(tid)];
            const size_t base = static_cast<size_t>(inst) * static_cast<size_t>(N);
            poly[base + static_cast<size_t>(2 * tid + 0)] = (limb & 0xFFFFu);
            poly[base + static_cast<size_t>(2 * tid + 1)] = (limb >> 16);
        }

        for (int i = DIGITS16 + tid; i < N; i += static_cast<int>(blockDim.x))
            poly[static_cast<size_t>(inst) * static_cast<size_t>(N) + static_cast<size_t>(i)] = 0;
    }

    __global__ void pack_b_u32_to_u64_poly16_kernel(const uint32_t *__restrict__ b,
                                                    Data64 *__restrict__ poly)
    {
        const int tid = static_cast<int>(threadIdx.x);
        if (tid < LEN_B)
        {
            const uint32_t limb = b[tid];
            poly[static_cast<size_t>(2 * tid + 0)] = (limb & 0xFFFFu);
            poly[static_cast<size_t>(2 * tid + 1)] = (limb >> 16);
        }

        for (int i = DIGITS16 + tid; i < N; i += static_cast<int>(blockDim.x))
            poly[static_cast<size_t>(i)] = 0;
    }

    __global__ void pointwise_mul_inplace_kernel(Data64 *__restrict__ a_hat,
                                                 const Data64 *__restrict__ b_hat,
                                                 Modulus<Data64> modulus,
                                                 int total_elems)
    {
        const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        if (idx >= total_elems)
            return;

        const int j = idx & (N - 1);
        const Data64 a = a_hat[static_cast<size_t>(idx)];
        const Data64 b = b_hat[static_cast<size_t>(j)];
        a_hat[static_cast<size_t>(idx)] = OPERATOR_GPU<Data64>::mult(a, b, modulus);
    }

    __global__ void digits16_to_limbs32_kernel(const Data64 *__restrict__ digits,
                                               uint32_t *__restrict__ out_limbs32,
                                               int batch_size)
    {
        const int inst = static_cast<int>(blockIdx.x);
        if (inst >= batch_size || threadIdx.x != 0)
            return;

        const Data64 *in = digits + static_cast<size_t>(inst) * static_cast<size_t>(N);
        uint32_t *out = out_limbs32 + static_cast<size_t>(inst) * static_cast<size_t>(LEN_RESULT);

        Data64 carry = 0;
        for (int k = 0; k < LEN_RESULT; k++)
        {
            Data64 t0 = in[static_cast<size_t>(2 * k + 0)] + carry;
            const uint32_t d0 = static_cast<uint32_t>(t0 & 0xFFFFu);
            carry = (t0 >> 16);

            Data64 t1 = in[static_cast<size_t>(2 * k + 1)] + carry;
            const uint32_t d1 = static_cast<uint32_t>(t1 & 0xFFFFu);
            carry = (t1 >> 16);

            out[static_cast<size_t>(k)] = d0 | (d1 << 16);
        }
    }
}

extern "C" int tc_mult_gpuntt_u32(const uint32_t *d_a, const uint32_t *d_b, uint32_t *d_result, size_t n)
{
    if (d_a == nullptr || d_b == nullptr || d_result == nullptr || n == 0)
        return TC_MULT_STATUS_INVALID_ARGUMENT;

    static bool inited = false;
    static Modulus<Data64> modulus;
    static Ninverse<Data64> n_inv = 0;
    static Root<Data64> *d_root_fwd = nullptr;
    static Root<Data64> *d_root_inv = nullptr;
    static Data64 *d_b_hat = nullptr;
    static const uint32_t *last_b_ptr = nullptr;
    static Data64 *d_a_poly = nullptr;

    if (!inited)
    {
        gpuntt::NTTParameters<Data64> params(N_POWER, gpuntt::ReductionPolynomial::X_N_minus);
        modulus = params.modulus;
        n_inv = params.n_inv;

        std::vector<Root<Data64>> table_fwd = params.gpu_root_of_unity_table_generator(params.forward_root_of_unity_table);
        cudaError_t err = cudaMalloc(&d_root_fwd, table_fwd.size() * sizeof(Root<Data64>));
        if (err != cudaSuccess)
            return cuda_status(err);
        err = cudaMemcpy(d_root_fwd, table_fwd.data(), table_fwd.size() * sizeof(Root<Data64>), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
            return cuda_status(err);

        std::vector<Root<Data64>> table_inv = params.gpu_root_of_unity_table_generator(params.inverse_root_of_unity_table);
        err = cudaMalloc(&d_root_inv, table_inv.size() * sizeof(Root<Data64>));
        if (err != cudaSuccess)
            return cuda_status(err);
        err = cudaMemcpy(d_root_inv, table_inv.data(), table_inv.size() * sizeof(Root<Data64>), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
            return cuda_status(err);

        err = cudaMalloc(&d_b_hat, static_cast<size_t>(N) * sizeof(Data64));
        if (err != cudaSuccess)
            return cuda_status(err);
        err = cudaMalloc(&d_a_poly, static_cast<size_t>(MAX_BATCH) * static_cast<size_t>(N) * sizeof(Data64));
        if (err != cudaSuccess)
            return cuda_status(err);

        inited = true;
    }

    if (d_b != last_b_ptr)
    {
        pack_b_u32_to_u64_poly16_kernel<<<1, THREADS_PACK>>>(d_b, d_b_hat);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            return cuda_status(err);

        gpuntt::ntt_configuration<Data64> cfg_fwd = {
            .n_power = N_POWER,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_minus,
            .zero_padding = false,
            .mod_inverse = 0,
            .stream = 0};

        gpuntt::GPU_NTT_Inplace(d_b_hat, d_root_fwd, modulus, cfg_fwd, 1);
        err = cudaGetLastError();
        if (err != cudaSuccess)
            return cuda_status(err);
        last_b_ptr = d_b;
    }

    gpuntt::ntt_configuration<Data64> cfg_ntt = {
        .n_power = N_POWER,
        .ntt_type = gpuntt::FORWARD,
        .ntt_layout = gpuntt::PerPolynomial,
        .reduction_poly = gpuntt::ReductionPolynomial::X_N_minus,
        .zero_padding = false,
        .mod_inverse = 0,
        .stream = 0};

    gpuntt::ntt_configuration<Data64> cfg_intt = {
        .n_power = N_POWER,
        .ntt_type = gpuntt::INVERSE,
        .ntt_layout = gpuntt::PerPolynomial,
        .reduction_poly = gpuntt::ReductionPolynomial::X_N_minus,
        .zero_padding = false,
        .mod_inverse = n_inv,
        .stream = 0};

    for (size_t base = 0; base < n; base += static_cast<size_t>(MAX_BATCH))
    {
        const int batch = static_cast<int>(std::min(static_cast<size_t>(MAX_BATCH), n - base));

        pack_a_u32_to_u64_poly16_kernel<<<batch, THREADS_PACK>>>(
            d_a + base * static_cast<size_t>(LEN_A), d_a_poly, batch);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            return cuda_status(err);

        gpuntt::GPU_NTT_Inplace(d_a_poly, d_root_fwd, modulus, cfg_ntt, batch);
        err = cudaGetLastError();
        if (err != cudaSuccess)
            return cuda_status(err);

        const int total = batch * N;
        const int threads = 256;
        const int blocks = (total + threads - 1) / threads;
        pointwise_mul_inplace_kernel<<<blocks, threads>>>(d_a_poly, d_b_hat, modulus, total);
        err = cudaGetLastError();
        if (err != cudaSuccess)
            return cuda_status(err);

        gpuntt::GPU_INTT_Inplace(d_a_poly, d_root_inv, modulus, cfg_intt, batch);
        err = cudaGetLastError();
        if (err != cudaSuccess)
            return cuda_status(err);

        digits16_to_limbs32_kernel<<<batch, 1>>>(d_a_poly, d_result + base * static_cast<size_t>(LEN_RESULT), batch);
        err = cudaGetLastError();
        if (err != cudaSuccess)
            return cuda_status(err);
    }

    return cuda_status(cudaDeviceSynchronize());
}
