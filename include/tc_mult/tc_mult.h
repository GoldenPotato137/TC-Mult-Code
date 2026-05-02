#pragma once

#include <stddef.h>
#include <stdint.h>

#ifndef TC_MULT_WORDS_A
#define TC_MULT_WORDS_A 192
#endif

#ifndef TC_MULT_WORDS_B
#define TC_MULT_WORDS_B TC_MULT_WORDS_A
#endif

#define TC_MULT_RESULT_WORDS (TC_MULT_WORDS_A + TC_MULT_WORDS_B)
#define TC_MULT_K_BYTES (TC_MULT_WORDS_A * 4)
#define TC_MULT_N_BYTES (TC_MULT_K_BYTES * 2)
#define TC_MULT_GMEM_B_BYTES (TC_MULT_K_BYTES * TC_MULT_N_BYTES)
#define TC_MULT_BLOCK_M 64

#ifdef __cplusplus
extern "C" {
#endif

typedef enum tc_mult_status_t {
    TC_MULT_STATUS_SUCCESS = 0,
    TC_MULT_STATUS_INVALID_ARGUMENT = 1,
    TC_MULT_STATUS_CUDA_ERROR = 2
} tc_mult_status_t;

/*
 * All multiplication APIs expect CUDA device pointers:
 *   d_a      : n big integers, each TC_MULT_WORDS_A 32-bit limbs.
 *   d_b      : one big integer, TC_MULT_WORDS_B 32-bit limbs.
 *   d_result : n products, each TC_MULT_RESULT_WORDS 32-bit limbs.
 *
 * TC-Mult optimized APIs require n to be a multiple of TC_MULT_BLOCK_M.
 */
int tc_mult_base_u32(const uint32_t *d_a, const uint32_t *d_b, uint32_t *d_result, size_t n);
int tc_mult_u32(const uint32_t *d_a, const uint32_t *d_b, uint32_t *d_result, size_t n);
int tc_mult_gmem_b_u32(const uint32_t *d_a, const uint8_t *d_gmem_b, uint32_t *d_result, size_t n);
int tc_mult_wo_perm_u32(const uint32_t *d_a, const uint32_t *d_b, uint32_t *d_result, size_t n);
int tc_mult_cgbn_u32(const uint32_t *d_a, const uint32_t *d_b, uint32_t *d_result, size_t n);
int tc_mult_gpuntt_u32(const uint32_t *d_a, const uint32_t *d_b, uint32_t *d_result, size_t n);

uint64_t tc_mult_hash_u32(const uint32_t *d_result, size_t n);
uint64_t tc_mult_hash_range_u32(const uint32_t *d_result, size_t n, uint32_t start_word, uint32_t word_count);

/* Host-side deterministic data helpers used by the examples and verifier. */
void tc_mult_fill_a_words(uint32_t *h_a, size_t n);
void tc_mult_fill_b_words(uint32_t *h_b);
void tc_mult_fill_a_with_constant_words(uint32_t *h_a, size_t n, const uint32_t *h_src);
void tc_mult_build_gmem_b_u8_from_b_words(uint8_t *h_gmem_b, const uint32_t *h_b);

#ifdef __cplusplus
}
#endif
