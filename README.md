# TC-Mult-Code

This repository contains the artifact package for the TC-Mult big-integer multiplication paper.

During review, TC-Mult is provided as a prebuilt binary library. The TC-Mult source code will be released after the review process is complete.

The public part of this package includes:

- `include/tc_mult/tc_mult.h`: stable C API.
- `src/base/`: public baseline source code.
- `examples/`: minimal invocation, verification, and benchmark examples.
- `lib/`: prebuilt TC-Mult binary libraries for review.
- `third_party/`: baseline submodules for CGBN and GPU-NTT.

## Included methods

The review binary exports the methods used in the paper:

| API | Meaning |
| --- | --- |
| `tc_mult_u32` | main TC-Mult implementation |
| `tc_mult_gmem_b_u32` | TC-Mult variant that reads the pre-packed B matrix from global memory |
| `tc_mult_wo_perm_u32` | TC-Mult w/o permutation variant, using the shared-memory carry path |
| `tc_mult_base_u32` | public source baseline |
| `tc_mult_cgbn_u32` | CGBN baseline |
| `tc_mult_gpuntt_u32` | GPU-NTT baseline |

## Requirements

- CUDA Toolkit with `nvcc`
- CMake >= 3.19
- C++20 compiler
- NVIDIA GPU supporting `sm_80` or `sm_89`

The binary libraries are built with CUDA architectures `80;89` and are provided for `TC_MULT_WORDS_A=TC_MULT_WORDS_B` equal to `64`, `128`, `192`, and `256`.

## Build

From this directory:

```bash
git submodule update --init --recursive #init submodules (CGBN/GPU-NTT)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="80;89"
cmake --build build --parallel
```

The default word size is `192`. To select another packaged binary, configure with matching word counts:

```bash
cmake -S . -B build-w64  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="80;89" -DTC_MULT_WORDS_A=64  -DTC_MULT_WORDS_B=64
cmake -S . -B build-w128 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="80;89" -DTC_MULT_WORDS_A=128 -DTC_MULT_WORDS_B=128
cmake -S . -B build-w256 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="80;89" -DTC_MULT_WORDS_A=256 -DTC_MULT_WORDS_B=256
```

## Run examples

```bash
./build/example_base 64
./build/example_tc_mult 64
./build/verify_all 64
./build/benchmark
```

`n` is the number of independent multiplications. TC-Mult optimized methods require `n` to be a positive multiple of `64`.

The benchmark example uses CUDA events and defaults to `n=2^20`:

```bash
./build/benchmark           # n = 2^20
./build/benchmark 1048576   # explicit n
```

It reports average runtime in milliseconds and throughput in MOp/s. The benchmark includes TC-Mult methods and all packaged baselines: `base`, `cgbn`, and `gpuntt`.

## C API

Include:

```c
#include "tc_mult/tc_mult.h"
```

All multiplication functions expect CUDA device pointers:

```c
int tc_mult_base_u32(const uint32_t *d_a,
                     const uint32_t *d_b,
                     uint32_t *d_result,
                     size_t n);

int tc_mult_u32(const uint32_t *d_a,
                const uint32_t *d_b,
                uint32_t *d_result,
                size_t n);

int tc_mult_gmem_b_u32(const uint32_t *d_a,
                       const uint8_t *d_gmem_b,
                       uint32_t *d_result,
                       size_t n);

int tc_mult_wo_perm_u32(const uint32_t *d_a,
                        const uint32_t *d_b,
                        uint32_t *d_result,
                        size_t n);

int tc_mult_cgbn_u32(const uint32_t *d_a,
                     const uint32_t *d_b,
                     uint32_t *d_result,
                     size_t n);

int tc_mult_gpuntt_u32(const uint32_t *d_a,
                       const uint32_t *d_b,
                       uint32_t *d_result,
                       size_t n);
```

Data layout:

- `d_a`: `n * TC_MULT_WORDS_A` 32-bit limbs.
- `d_b`: `TC_MULT_WORDS_B` 32-bit limbs.
- `d_result`: `n * TC_MULT_RESULT_WORDS` 32-bit limbs.
- `d_gmem_b`: `TC_MULT_GMEM_B_BYTES` bytes, prepared on host with `tc_mult_build_gmem_b_u8_from_b_words` and copied to device.

Helper functions:

```c
void tc_mult_fill_a_words(uint32_t *h_a, size_t n);
void tc_mult_fill_b_words(uint32_t *h_b);
void tc_mult_build_gmem_b_u8_from_b_words(uint8_t *h_gmem_b, const uint32_t *h_b);
uint64_t tc_mult_hash_u32(const uint32_t *d_result, size_t n);
```

## Minimal calling example

```cpp
#include <vector>
#include <cuda_runtime_api.h>
#include "tc_mult/tc_mult.h"

int main() {
    size_t n = 64;
    std::vector<uint32_t> h_a(n * TC_MULT_WORDS_A);
    std::vector<uint32_t> h_b(TC_MULT_WORDS_B);
    tc_mult_fill_a_words(h_a.data(), n);
    tc_mult_fill_b_words(h_b.data());

    uint32_t *d_a = nullptr, *d_b = nullptr, *d_result = nullptr;
    cudaMalloc(&d_a, h_a.size() * sizeof(uint32_t));
    cudaMalloc(&d_b, h_b.size() * sizeof(uint32_t));
    cudaMalloc(&d_result, n * TC_MULT_RESULT_WORDS * sizeof(uint32_t));
    cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

    tc_mult_u32(d_a, d_b, d_result, n);
    uint64_t hash = tc_mult_hash_u32(d_result, n);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    return hash == 0 ? 1 : 0;
}
```

See `examples/verify_all.cu` for a complete correctness check against the public base implementation.
