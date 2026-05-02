#include "common.cuh"

namespace
{
    void print_check(const char *name, uint64_t hash, uint64_t ref_hash)
    {
        bool ok = hash == ref_hash;
        std::printf("%-18s hash=%llu [%s]\n", name, static_cast<unsigned long long>(hash), ok ? "OK" : "FAIL");
        if (!ok)
            std::exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv)
{
    size_t n = parse_n(argc, argv);
    device_inputs data(n);
    data.ensure_gmem_b();

    data.clear_result();
    require_success("tc_mult_base_u32", tc_mult_base_u32(data.d_a, data.d_b, data.d_result, n));
    uint64_t ref_hash = tc_mult_hash_u32(data.d_result, n);

    data.clear_result();
    require_success("tc_mult_u32", tc_mult_u32(data.d_a, data.d_b, data.d_result, n));
    print_check("tc_mult", tc_mult_hash_u32(data.d_result, n), ref_hash);

    data.clear_result();
    require_success("tc_mult_gmem_b_u32", tc_mult_gmem_b_u32(data.d_a, data.d_gmem_b, data.d_result, n));
    print_check("tc_mult_gmem_b", tc_mult_hash_u32(data.d_result, n), ref_hash);

    data.clear_result();
    require_success("tc_mult_wo_perm_u32", tc_mult_wo_perm_u32(data.d_a, data.d_b, data.d_result, n));
    print_check("tc_mult_wo_perm", tc_mult_hash_u32(data.d_result, n), ref_hash);

    std::printf("base              hash=%llu [REFERENCE]\n", static_cast<unsigned long long>(ref_hash));
    std::printf("verification passed for n=%zu, words=%d\n", n, TC_MULT_WORDS_A);
    return 0;
}
