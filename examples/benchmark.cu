#include "common.cuh"

namespace
{
    constexpr size_t DEFAULT_N = 1u << 20;
    constexpr int DEFAULT_ITERS = 10;
    constexpr int DEFAULT_WARMUP = 3;
    constexpr int BASELINE_ITERS = 3;
    constexpr int BASELINE_WARMUP = 1;

    using benchmark_fn = int (*)(device_inputs &);

    int run_tc_mult(device_inputs &data)
    {
        return tc_mult_u32(data.d_a, data.d_b, data.d_result, data.n);
    }

    int run_tc_mult_gmem_b(device_inputs &data)
    {
        return tc_mult_gmem_b_u32(data.d_a, data.d_gmem_b, data.d_result, data.n);
    }

    int run_tc_mult_wo_perm(device_inputs &data)
    {
        return tc_mult_wo_perm_u32(data.d_a, data.d_b, data.d_result, data.n);
    }

    int run_base(device_inputs &data)
    {
        return tc_mult_base_u32(data.d_a, data.d_b, data.d_result, data.n);
    }

    int run_cgbn(device_inputs &data)
    {
        return tc_mult_cgbn_u32(data.d_a, data.d_b, data.d_result, data.n);
    }

    int run_gpuntt(device_inputs &data)
    {
        return tc_mult_gpuntt_u32(data.d_a, data.d_b, data.d_result, data.n);
    }

    struct benchmark_case
    {
        const char *name;
        benchmark_fn fn;
        int warmup;
        int iters;
    };

    double benchmark_one(device_inputs &data, const benchmark_case &bench)
    {
        for (int i = 0; i < bench.warmup; i++)
        {
            data.clear_result();
            require_success(bench.name, bench.fn(data));
        }

        cudaEvent_t start;
        cudaEvent_t stop;
        TC_MULT_EXAMPLE_CUDA_CHECK(cudaEventCreate(&start));
        TC_MULT_EXAMPLE_CUDA_CHECK(cudaEventCreate(&stop));

        data.clear_result();
        TC_MULT_EXAMPLE_CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < bench.iters; i++)
            require_success(bench.name, bench.fn(data));
        TC_MULT_EXAMPLE_CUDA_CHECK(cudaEventRecord(stop));
        TC_MULT_EXAMPLE_CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        TC_MULT_EXAMPLE_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        TC_MULT_EXAMPLE_CUDA_CHECK(cudaEventDestroy(start));
        TC_MULT_EXAMPLE_CUDA_CHECK(cudaEventDestroy(stop));

        return static_cast<double>(elapsed_ms) / static_cast<double>(bench.iters);
    }

    size_t parse_benchmark_n(int argc, char **argv)
    {
        size_t n = DEFAULT_N;
        if (argc > 1)
            n = static_cast<size_t>(std::strtoull(argv[1], nullptr, 10));
        if (n == 0 || n % TC_MULT_BLOCK_M != 0)
        {
            std::fprintf(stderr, "n must be a positive multiple of %d\n", TC_MULT_BLOCK_M);
            std::exit(EXIT_FAILURE);
        }
        return n;
    }
}

int main(int argc, char **argv)
{
    size_t n = parse_benchmark_n(argc, argv);
    device_inputs data(n);
    data.ensure_gmem_b();

    benchmark_case cases[] = {
        {"tc_mult", run_tc_mult, DEFAULT_WARMUP, DEFAULT_ITERS},
        {"tc_mult_gmem_b", run_tc_mult_gmem_b, DEFAULT_WARMUP, DEFAULT_ITERS},
        {"tc_mult_wo_perm", run_tc_mult_wo_perm, DEFAULT_WARMUP, DEFAULT_ITERS},
        {"base", run_base, BASELINE_WARMUP, BASELINE_ITERS},
        {"cgbn", run_cgbn, BASELINE_WARMUP, BASELINE_ITERS},
        {"gpuntt", run_gpuntt, BASELINE_WARMUP, BASELINE_ITERS},
    };

    std::printf("TC-Mult benchmark: n=%zu, words=%d\n", n, TC_MULT_WORDS_A);
    std::printf("%-18s %8s %8s %14s %14s\n", "method", "warmup", "iters", "time_ms", "MOp/s");

    for (const auto &bench : cases)
    {
        double time_ms = benchmark_one(data, bench);
        double mops = static_cast<double>(n) / time_ms / 1000.0;
        std::printf("%-18s %8d %8d %14.6f %14.6f\n", bench.name, bench.warmup, bench.iters, time_ms, mops);
    }

    return 0;
}
