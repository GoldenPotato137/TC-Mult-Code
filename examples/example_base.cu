#include "common.cuh"

int main(int argc, char **argv)
{
    size_t n = parse_n(argc, argv);
    device_inputs data(n);

    data.clear_result();
    require_success("tc_mult_base_u32", tc_mult_base_u32(data.d_a, data.d_b, data.d_result, n));
    uint64_t hash = tc_mult_hash_u32(data.d_result, n);

    std::printf("base: n=%zu words=%d hash=%llu\n", n, TC_MULT_WORDS_A, static_cast<unsigned long long>(hash));
    return 0;
}
