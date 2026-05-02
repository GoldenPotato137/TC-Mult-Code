#include "common.cuh"

int main(int argc, char **argv)
{
    size_t n = parse_n(argc, argv);
    device_inputs data(n);
    data.ensure_gmem_b();

    data.clear_result();
    require_success("tc_mult_u32", tc_mult_u32(data.d_a, data.d_b, data.d_result, n));
    uint64_t tc_mult_hash = tc_mult_hash_u32(data.d_result, n);

    data.clear_result();
    require_success("tc_mult_gmem_b_u32", tc_mult_gmem_b_u32(data.d_a, data.d_gmem_b, data.d_result, n));
    uint64_t gmem_b_hash = tc_mult_hash_u32(data.d_result, n);

    data.clear_result();
    require_success("tc_mult_wo_perm_u32", tc_mult_wo_perm_u32(data.d_a, data.d_b, data.d_result, n));
    uint64_t wo_perm_hash = tc_mult_hash_u32(data.d_result, n);

    std::printf("tc_mult        : n=%zu words=%d hash=%llu\n", n, TC_MULT_WORDS_A, static_cast<unsigned long long>(tc_mult_hash));
    std::printf("tc_mult_gmem_b : n=%zu words=%d hash=%llu\n", n, TC_MULT_WORDS_A, static_cast<unsigned long long>(gmem_b_hash));
    std::printf("tc_mult_wo_perm: n=%zu words=%d hash=%llu\n", n, TC_MULT_WORDS_A, static_cast<unsigned long long>(wo_perm_hash));
    return 0;
}
