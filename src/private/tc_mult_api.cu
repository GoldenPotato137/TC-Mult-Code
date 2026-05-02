#include "tc_mult/tc_mult.h"

#include <cuda_runtime_api.h>

#define TC_MULT_CAT2(a, b) a##b
#define TC_MULT_CAT(a, b) TC_MULT_CAT2(a, b)
#define TC_MULT_IMPL_PREFIX TC_MULT_CAT(ten, sor_)
#define TC_MULT_IMPL_MAIN_NS TC_MULT_CAT(TC_MULT_IMPL_PREFIX, v2)
#define TC_MULT_IMPL_GMEM_B_NS TC_MULT_CAT(TC_MULT_IMPL_MAIN_NS, _gmem_b)
#define TC_MULT_IMPL_WO_PERM_NS TC_MULT_CAT(TC_MULT_IMPL_MAIN_NS, _smem_carry)

namespace TC_MULT_IMPL_MAIN_NS
{
    uint64_t calc(uint32_t *a, uint8_t *b, uint32_t *result, size_t n);
}

namespace TC_MULT_IMPL_GMEM_B_NS
{
    uint64_t calc(uint32_t *a, uint8_t *b, uint32_t *result, size_t n);
}

namespace TC_MULT_IMPL_WO_PERM_NS
{
    uint64_t calc(uint32_t *a, uint8_t *b, uint32_t *result, size_t n);
}

namespace
{
    int validate_tc_mult_args(const void *a, const void *b, const void *result, size_t n)
    {
        if (a == nullptr || b == nullptr || result == nullptr || n == 0)
            return TC_MULT_STATUS_INVALID_ARGUMENT;
        if (n % TC_MULT_BLOCK_M != 0)
            return TC_MULT_STATUS_INVALID_ARGUMENT;
        return TC_MULT_STATUS_SUCCESS;
    }

    int status_after_call()
    {
        cudaError_t err = cudaGetLastError();
        return err == cudaSuccess ? TC_MULT_STATUS_SUCCESS : TC_MULT_STATUS_CUDA_ERROR;
    }
}

extern "C" int tc_mult_u32(const uint32_t *d_a, const uint32_t *d_b, uint32_t *d_result, size_t n)
{
    int status = validate_tc_mult_args(d_a, d_b, d_result, n);
    if (status != TC_MULT_STATUS_SUCCESS)
        return status;

    TC_MULT_IMPL_MAIN_NS::calc(
        const_cast<uint32_t *>(d_a),
        reinterpret_cast<uint8_t *>(const_cast<uint32_t *>(d_b)),
        d_result,
        n);
    return status_after_call();
}

extern "C" int tc_mult_gmem_b_u32(const uint32_t *d_a, const uint8_t *d_gmem_b, uint32_t *d_result, size_t n)
{
    int status = validate_tc_mult_args(d_a, d_gmem_b, d_result, n);
    if (status != TC_MULT_STATUS_SUCCESS)
        return status;

    TC_MULT_IMPL_GMEM_B_NS::calc(
        const_cast<uint32_t *>(d_a),
        const_cast<uint8_t *>(d_gmem_b),
        d_result,
        n);
    return status_after_call();
}

extern "C" int tc_mult_wo_perm_u32(const uint32_t *d_a, const uint32_t *d_b, uint32_t *d_result, size_t n)
{
    int status = validate_tc_mult_args(d_a, d_b, d_result, n);
    if (status != TC_MULT_STATUS_SUCCESS)
        return status;

    TC_MULT_IMPL_WO_PERM_NS::calc(
        const_cast<uint32_t *>(d_a),
        reinterpret_cast<uint8_t *>(const_cast<uint32_t *>(d_b)),
        d_result,
        n);
    return status_after_call();
}
