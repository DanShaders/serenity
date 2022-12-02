/*
 * Copyright (c) 2022, Dan Klishch <danilklishch@gmail.com>
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <AK/BigIntBase.h>
#include <AK/SIMD.h>
#include <AK/UFixedBigInt.h>

namespace AK {

// TODO: update description

// ===== Multiplication =====
// Time complexity: O(n log n)
//
// On the algorithmic level, we do discrete Fourier transform over Z/modulus using mixed-radix
// Cooley-Tukey FFT. Radix (`1 << inner_iters`) is 64 except for the first iteration of the main
// loop (when it can be smaller in order to match `fft_length`). Multiplication by arbitrary
// twiddle factors (happening between main loop iterations) is done in ntt_multiply_arbitrary.
// It should be noted that we gather all of the required twiddle factors from twiddle_factors into
// local_factors/twiddle_buffer to improve cache locality.
// Twiddle factors of inner NTTs (of size 64 or less) are happen to be powers of 2. These NTTs are
// located in ntt_* (top two loops) and ntt_*_convolve (SIMD-unrollable inner loop).
//
// Very high-level description of NTT for this particular modulus can be found at
// https://cp4space.hatsya.com/2021/09/01/an-efficient-prime-for-number-theoretic-transforms/ .

constexpr size_t one_sz = 1;

constexpr u64 modulus = 0xffff'ffff'0000'0001ULL;

// These are `2 ** i`-th roots of the unity modulo `modulus`
constexpr u64 roots_of_unity[] = {
    18446744069414584321ULL,
    18446744069414584320ULL,
    281474976710656ULL,
    16777216ULL,
    4096ULL,
    64ULL,
    8ULL,
    2198989700608ULL,
    14041890976876060974ULL,
    14430643036723656017ULL,
    4440654710286119610ULL,
    8816101479115663336ULL,
    10974926054405199669ULL,
    1206500561358145487ULL,
    10930245224889659871ULL,
    3333600369887534767ULL,
    15893793146607301539ULL,
    14445062887364698470ULL,
    12687654034874795207ULL,
    4998280027294208343ULL,
    2386580402828090423ULL,
    14917392722903128661ULL,
    14689788312086715899ULL,
    14780543597221338054ULL,
    14670161887888854203ULL,
    17585967655187380175ULL,
    2561969345295687999ULL,
    3842056917760402699ULL,
    9792270020272749848ULL,
    7552600543241881301ULL,
    8315689427686264475ULL,
    7768485315656529096ULL,
    16334397945464290598ULL,
};

// clang-format off
constexpr size_t reversed_bits_64[] = {
    0, 32, 16, 48, 8,  40, 24, 56, 4, 36, 20, 52, 12, 44, 28, 60,
    2, 34, 18, 50, 10, 42, 26, 58, 6, 38, 22, 54, 14, 46, 30, 62,
    1, 33, 17, 49, 9,  41, 25, 57, 5, 37, 21, 53, 13, 45, 29, 61,
    3, 35, 19, 51, 11, 43, 27, 59, 7, 39, 23, 55, 15, 47, 31, 63,
};
// clang-format on

template<typename Func, size_t... shift_categories>
ALWAYS_INLINE void shift_dispatch(Func func)
{
    func.template operator()<shift_categories...>();
}

template<typename Func, size_t... shift_categories>
ALWAYS_INLINE void shift_dispatch(Func func, size_t s0, SameAs<size_t> auto... s)
{
    if (s0 == 0) {
        shift_dispatch<Func, shift_categories..., 0>(func, s...);
    } else if (s0 <= 32) {
        shift_dispatch<Func, shift_categories..., 1>(func, s...);
    } else if (s0 < 64) {
        shift_dispatch<Func, shift_categories..., 2>(func, s...);
    } else {
        shift_dispatch<Func, shift_categories..., 3>(func, s...);
    }
}

#if defined(AK_COMPILER_GCC) && ARCH(X86_64)
// GCC currently produces much worse code than clang does (and the following is the clang's output)
__attribute__((naked, noinline)) static u64 mod_reduce(u64, u64, u64)
{
    asm(R"(
        subq %rdx, %rdi
        movabsq $-4294967296, %rcx
        leaq 1(%rdi,%rcx), %rdx
        cmovaeq %rdi, %rdx
        movl %esi, %eax
        shlq $32, %rsi
        subq %rax, %rsi
        leaq (%rdx,%rsi), %rax
        cmpq %rcx, %rax
        movl $4294967295, %ecx
        leaq (%rax,%rcx), %rcx
        cmovaq %rcx, %rax
        addq %rdx, %rsi
        cmovbq %rcx, %rax
        retq
    )");
}
#else
// This is the only code from the article. I do not know how to write the following 10 lines in a
// different way, so the function is directly copied.
static u64 mod_reduce(u64 low, u64 middle, u64 high)
{
    u64 low2 = low - high;
    if (high > low)
        low2 += modulus;
    u64 product = middle << 32;
    product -= product >> 32;
    u64 result = low2 + product;
    if (result < product || result >= modulus)
        result -= modulus;
    return result;
}
#endif

// (a * b) % modulus
static u64 mod_mul(u64 a, u64 b)
{
    u128 c = AK::UFixedBigInt<64>(a).wide_multiply(b);
    return mod_reduce(c.low(), c.high() & 0xffff'ffffULL, c.high() >> 32);
}

// (a + b) % modulus
static u64 mod_add(u64 a, u64 b)
{
    return a + b - (b < modulus - a ? 0 : modulus);
}

// (a - b) % modulus
static u64 mod_sub(u64 a, u64 b)
{
    return a - b + (a < b ? modulus : 0);
}

template<size_t sc>
static u64 mod_shift(u64 y, u64 shift)
{
    if constexpr (sc == 0) {
        return y;
    } else if constexpr (sc == 1) {
        return mod_reduce(y << shift, y >> (64 - shift), 0);
    } else if constexpr (sc == 2) {
        return mod_reduce(y << shift, static_cast<u32>(y >> (64 - shift)), y >> (96 - shift));
    } else if constexpr (sc == 3) {
        return mod_reduce(0, y << (shift - 64), y >> (96 - shift));
    } else if constexpr (sc == 4) {
        if (shift == 0) {
            return mod_shift<0>(y, shift);
        } else if (shift <= 32) {
            return mod_shift<1>(y, shift);
        } else if (shift < 64) {
            return mod_shift<2>(y, shift);
        } else {
            return mod_shift<3>(y, shift);
        }
    } else {
        VERIFY_NOT_REACHED();
    }
}

template<SIMD::UnrollingMode>
ALWAYS_INLINE static void ntt_convolve(u64* a, size_t j, size_t part_len, size_t total_len, size_t shift)
{
    shift_dispatch([&]<size_t s> {
        for (size_t h = j; h < j + part_len; ++h) {
            auto x = a[h], y = mod_shift<s>(a[h + total_len], shift);
            a[h] = mod_add(x, y), a[h + total_len] = mod_sub(x, y);
        }
    },
        shift);
}

template<SIMD::UnrollingMode unrolling_mode>
ALWAYS_INLINE static void ntt_convolve2(u64* a, size_t j, size_t part_len, size_t shift1, size_t shift2, size_t shift3)
{
    ntt_convolve<unrolling_mode>(a, j, 2 * part_len, 2 * part_len, shift1);
    ntt_convolve<unrolling_mode>(a, j, part_len, part_len, shift2);
    ntt_convolve<unrolling_mode>(a, j + 2 * part_len, part_len, part_len, shift3);
}

template<SIMD::UnrollingMode unrolling_mode>
ALWAYS_INLINE static void ntt_convolve3(u64* a, size_t j, size_t part_len, size_t total_len, size_t shift1, size_t shift2, size_t shift3)
{
    ntt_convolve<unrolling_mode>(a, j, part_len, total_len, shift1);
    ntt_convolve<unrolling_mode>(a, j + 2 * total_len, part_len, total_len, shift1);
    ntt_convolve<unrolling_mode>(a, j, part_len, 2 * total_len, shift2);
    ntt_convolve<unrolling_mode>(a, j + total_len, part_len, 2 * total_len, shift3);
}

template<SIMD::UnrollingMode>
ALWAYS_INLINE static void ntt_multiply_arbitrary(size_t from, size_t to, size_t scale, u64* a, u64* local_factors)
{
    size_t scaling = local_factors[scale];
    size_t factor = 1;
    for (size_t i = from; i < to; ++i) {
        a[i] = mod_mul(a[i], factor);
        factor = mod_mul(factor, scaling);
    }
}

template<SIMD::UnrollingMode>
ALWAYS_INLINE static void ntt_multiply_arbitrary2(size_t from, size_t to, size_t scale, size_t idx_mask, u64* a, u64* local_factors)
{
    size_t factor_idx = 0;

    for (size_t i = from; i < to; ++i) {
        a[i] = mod_mul(a[i], local_factors[factor_idx]);
        factor_idx += scale;
        factor_idx &= idx_mask;
    }
}

template<SIMD::UnrollingMode>
ALWAYS_INLINE static void ntt_multiply_constant(u64* a, size_t from, size_t to, u64 value)
{
    for (size_t i = from; i < to; ++i)
        a[i] = mod_mul(a[i], value);
}

template<SIMD::UnrollingMode unrolling_mode>
static void nttf(size_t k, size_t n, size_t n2, u64* a, u64* twiddle_start, u64* twiddle_sparse, u64* twiddle_buffer, NativeWord* reversed_idx)
{
    for (size_t outer_log_len = 0; outer_log_len < k;) {
        size_t inner_iters = outer_log_len || k % 6 == 0 ? 6 : k % 6;
        size_t part_len = one_sz << outer_log_len;
        size_t parts = one_sz << inner_iters;
        size_t shift = k - inner_iters - outer_log_len;

        // Unlike in `nttr`, I don't know what we've computed at this point. :)
        // We should have done something with blocks of `part_len` numbers in a[reversed_idx[...]].

        // Multiply by twiddle factors
        if (outer_log_len) {
            size_t total_len = one_sz << (k - outer_log_len);
            size_t inner_len = one_sz << shift;

            if (inner_len == 1) {
                for (size_t i = 0; i < n2; i += parts) {
                    size_t offset_in_part = reversed_idx[(i & (n - 1)) >> 6] >> 6;
                    ntt_multiply_arbitrary<unrolling_mode>(i, i + parts, offset_in_part, a, twiddle_start);
                }
            } else {
                u64* local_factors = twiddle_sparse;
                if (shift > 6) {
                    local_factors = twiddle_buffer;
                    for (size_t i = 0; i < (n >> shift); ++i)
                        local_factors[i] = twiddle_sparse[i << (shift - 6)];
                }

                for (size_t i = 0; i < n2; i += total_len) {
                    size_t offset_in_part = reversed_idx[(i & (n - 1)) >> 12] >> 12 & (part_len - 1);
                    if (!offset_in_part)
                        continue;

                    for (size_t j = i, idx = 0; j < i + total_len; j += inner_len, (idx += offset_in_part) &= (n >> shift) - 1)
                        if (idx)
                            ntt_multiply_constant<unrolling_mode>(a, j, j + inner_len, local_factors[idx]);
                }
            }
        }

        // Do NTTs of size `parts`
        for (size_t log_len = 0; log_len < inner_iters; ++log_len) {
            size_t total_len = one_sz << (k - outer_log_len - log_len - 1);
            if (inner_iters - log_len >= 2 && total_len >= 8) {
                size_t total_len2 = one_sz << (k - outer_log_len - log_len - 2);

                for (size_t i = 0; i < n2; i += 2 * total_len) {
                    size_t shift1 = 3 * (reversed_bits_64[i >> max(0, int(k) - int(outer_log_len) - 6) & 63] << (6 - log_len - 1) & 63);
                    size_t shift2 = 3 * (reversed_bits_64[i >> max(0, int(k) - int(outer_log_len) - 6) & 63] << (6 - log_len - 2) & 63);
                    size_t shift3 = 3 * (reversed_bits_64[(i + total_len) >> max(0, int(k) - int(outer_log_len) - 6) & 63] << (6 - log_len - 2) & 63);
                    ntt_convolve2<unrolling_mode>(a, i, total_len2, shift1, shift2, shift3);
                }

                ++log_len;
            } else {
                if (total_len == 2) [[likely]] {
                    for (size_t i = 0; i < n2; i += 4) {
                        size_t shift2 = reversed_bits_64[i & 63]; // [0; 16)
                        size_t shift3 = shift2 + 16;              // [16; 32)
                        size_t shift1 = shift2 << 1;

                        auto x = a[i], y = mod_shift<4>(a[i + 2], 3 * shift1);
                        a[i] = mod_add(x, y), a[i + 2] = mod_sub(x, y);
                        x = a[i + 1], y = mod_shift<4>(a[i + 3], 3 * shift1);
                        a[i + 1] = mod_add(x, y), a[i + 3] = mod_sub(x, y);

                        x = a[i], y = mod_shift<4>(a[i + 1], 3 * shift2);
                        a[i] = mod_add(x, y), a[i + 1] = mod_sub(x, y);

                        x = a[i + 2], y = mod_shift<4>(a[i + 3], 3 * shift3);
                        a[i + 2] = mod_add(x, y), a[i + 3] = mod_sub(x, y);
                    }
                    ++log_len;
                } else {
                    // This should not be reachable at all for sufficiently large k (kept for debug purposes)
                    for (size_t i = 0; i < n2; i += 2 * total_len) {
                        size_t shift = 3 * (reversed_bits_64[i >> max(0, int(k) - int(outer_log_len) - 6) & 63] << (6 - log_len - 1) & 63);
                        ntt_convolve<unrolling_mode>(a, i, total_len, total_len, shift);
                    }
                }
            }
        }

        outer_log_len += inner_iters;
    }
}

template<SIMD::UnrollingMode unrolling_mode>
static void nttr(size_t k, size_t n, size_t n2, u64* a, u64* twiddle_start, u64* twiddle_sparse, u64* twiddle_buffer)
{
    for (size_t outer_log_len = 0; outer_log_len < k;) {
        size_t inner_iters = outer_log_len || k % 6 == 0 ? 6 : k % 6;
        size_t part_len = one_sz << outer_log_len;
        size_t shift = k - inner_iters - outer_log_len;

        // We've computed NTT of each consecutive block of `part_len` numbers.
        // Now we want to merge `parts` consecutive NTTs into one NTT of size `block_len`.

        // Multiply by twiddle factors
        if (outer_log_len) {
            if (shift) {
                u64* local_factors = twiddle_sparse;
                if (shift > 6) {
                    local_factors = twiddle_buffer;
                    for (size_t i = 0; i < (n >> shift); ++i)
                        local_factors[i] = twiddle_sparse[i << (shift - 6)];
                }

                for (size_t part_offset = 0; part_offset < n2; part_offset += part_len) {
                    size_t part_in_block = reversed_bits_64[part_offset >> outer_log_len & 63];
                    if (!part_in_block)
                        continue;

                    // Since twiddle_buffer is usually small enough to fit in the cache, it's faster
                    // to scatter-gather values from it rather than recomputing them.
                    ntt_multiply_arbitrary2<unrolling_mode>(part_offset, part_offset + part_len, part_in_block, (n >> shift) - 1, a, local_factors);
                }
            } else {
                for (size_t part_offset = 0; part_offset < n2; part_offset += part_len) {
                    size_t part_in_block = reversed_bits_64[part_offset >> outer_log_len & 63];
                    if (!part_in_block)
                        continue;

                    ntt_multiply_arbitrary<unrolling_mode>(part_offset, part_offset + part_len, part_in_block, a, twiddle_start);
                }
            }
        }

        // Do NTTs of size `parts`
        for (size_t log_len = 0; log_len < inner_iters; ++log_len) {
            size_t len = one_sz << log_len;
            size_t total_len = len << outer_log_len;

            if (inner_iters - log_len >= 2) {
                for (size_t i = 0; i < n2; i += 4 * total_len) {
                    for (size_t j = i; j < i + total_len; j += part_len) {
                        size_t shift1 = 3 * ((j - i) >> outer_log_len << (6 - log_len - 1) & 63);
                        size_t shift2 = 3 * ((j - i) >> outer_log_len << (5 - log_len - 1) & 63);
                        size_t shift3 = 3 * ((j - i + total_len) >> outer_log_len << (5 - log_len - 1) & 63);
                        ntt_convolve3<unrolling_mode>(a, j, part_len, total_len, shift1, shift2, shift3);
                    }
                }
                ++log_len;
            } else {
                for (size_t i = 0; i < n2; i += 2 * total_len) {
                    for (size_t j = i; j < i + total_len; j += part_len) {
                        size_t shift = 3 * ((j - i) >> outer_log_len << (6 - log_len - 1) & 63);
                        ntt_convolve<unrolling_mode>(a, j, part_len, total_len, shift);
                    }
                }
            }
        }

        outer_log_len += inner_iters;
    }
}

template<SIMD::UnrollingMode unrolling_mode>
static void storage_mul_ntt_using(NativeWord const* data1, size_t size1, NativeWord const* data2, size_t size2, NativeWord* result, size_t size, NativeWord* buffer)
{
    auto allocate_u64 = [&](size_t amount) {
        u64* result = reinterpret_cast<u64*>(buffer);
        buffer += (word_size == 32 ? 2 : 1) * amount;
        return result;
    };

    auto allocate_native = [&](size_t amount) {
        NativeWord* result = reinterpret_cast<NativeWord*>(buffer);
        buffer += amount;
        return result;
    };

    size_t full_result_length = size1 + size2;

    // FIXME: Find out a way to split multiplication of extremely huge numbers into smaller
    //        multiplications to lift this restriction and reduce memory usage.
    size_t full_result_length_in_u32 = full_result_length * (word_size == 32 ? 1 : 2);
    VERIFY(full_result_length_in_u32 < (one_sz << 31)); // The exact bound is `modulus / 2 ** 33`.

    size_t k = sizeof(size_t) * 8 - count_leading_zeroes(full_result_length) - 1;
    if (full_result_length & (full_result_length - 1))
        ++k;
    k += (word_size == 32 ? 1 : 2);
    size_t n = one_sz << k;
    VERIFY(k >= 4);

    // Count twiddle factors
    u64 root = roots_of_unity[k];

    u64* twiddle_start = nullptr;
    u64* twiddle_sparse = nullptr;

    if (n > 64) {
        twiddle_start = allocate_u64(max(64u, n >> 6));
        twiddle_sparse = allocate_u64(n >> 6);

        twiddle_start[0] = 1;
        twiddle_sparse[0] = 1;

        for (size_t i = 1; i < max(64u, n >> 6); ++i)
            twiddle_start[i] = mod_mul(twiddle_start[i - 1], root);

        u64 scaling = roots_of_unity[k - 6];
        for (size_t i = 1; i < (n >> 6); ++i)
            twiddle_sparse[i] = mod_mul(twiddle_sparse[i - 1], scaling);
    } else {
        twiddle_start = allocate_u64(n);

        for (size_t i = 1; i < n; ++i)
            twiddle_start[i] = mod_mul(twiddle_start[i - 1], root);
    }

    SIMD::align_up(buffer, unrolling_mode);

    // Split arguments into u16 words
    auto split = [&](NativeWord const* data, size_t size, u64* operand) {
        if constexpr (word_size == 32) {
            for (size_t i = 0; i < size; ++i) {
                operand[2 * i] = data[i] & 0xffff;
                operand[2 * i + 1] = data[i] >> 16;
            }
            memset(operand + 2 * size, 0, sizeof(u64) * (n - 2 * size));
        } else {
            for (size_t i = 0; i < size; ++i) {
                u64 current = data[i];
                operand[4 * i] = static_cast<u16>(current);
                operand[4 * i + 1] = static_cast<u16>(current >> 16);
                operand[4 * i + 2] = static_cast<u16>(current >> 32);
                operand[4 * i + 3] = static_cast<u16>(current >> 48);
            }
            memset(operand + 4 * size, 0, sizeof(u64) * (n - 4 * size));
        }
    };

    u64* operand1 = reinterpret_cast<u64*>(buffer);
    split(data1, size1, operand1);
    buffer += (word_size == 32 ? 2 : 1) * n; // u64[n]

    u64* operand2 = reinterpret_cast<u64*>(buffer);
    split(data2, size2, operand2);
    buffer += (word_size == 32 ? 2 : 1) * n; // u64[n]

    // Count reversed bit representations of indexes
    NativeWord* reversed_idx = allocate_native(n >> 6);
    if (n >= 64) {
        reversed_idx[0] = 0;
        for (size_t i = 0; i < k - 6; ++i)
            reversed_idx[one_sz << i] = one_sz << (k - i - 1);
        for (size_t i = 1; i < (n >> 6); ++i)
            reversed_idx[i] = reversed_idx[i & (i - 1)] ^ reversed_idx[i ^ (i & (i - 1))];
    }

    // "Allocate" twiddle factors buffer
    u64* twiddle_buffer = reinterpret_cast<u64*>(buffer);

    // Multiplication time has come.
    nttf<unrolling_mode>(k, n, 2 * n, operand1, twiddle_start, twiddle_sparse, twiddle_buffer, reversed_idx);
    for (size_t i = 0; i < n; ++i)
        operand1[i] = mod_mul(operand1[i], operand2[i]);
    nttr<unrolling_mode>(k, n, n, operand1, twiddle_start, twiddle_sparse, twiddle_buffer);
    for (size_t i = 1; i < n / 2; ++i)
        swap(operand1[i], operand1[n - i]);

    u64 modular_inverse_of_n = modulus - (modulus >> k); // (modular_inverse_of_n * n) % modulus == 1
    for (size_t i = 0; i < n; ++i)
        operand1[i] = mod_mul(operand1[i], modular_inverse_of_n);

    // Copy everything they want from us to the result
    constexpr size_t scale = word_size / 16;
    memset(result, 0, sizeof(NativeWord) * size);

    u64 carry = 0;
    for (size_t i = 0; i < min(n, size * scale); ++i) {
        bool local_carry = false;

        if constexpr (word_size == 64) {
            add_words(carry, operand1[i], carry, local_carry);
        } else {
            NativeWord low_carry, high_carry;
            add_words(carry, operand1[i], low_carry, local_carry);
            add_words(carry >> 32, operand1[i] >> 32, high_carry, local_carry);
            carry = (static_cast<u64>(high_carry) << 32) + low_carry;
        }

        result[i / scale] |= (carry & 0xffff) << (i % scale * 16);
        carry >>= 16;
        carry += static_cast<u64>(local_carry) << 48;
    }
}

void storage_mul_smart(NativeWord const* data1, size_t size1, NativeWord const* data2, size_t size2, NativeWord* result, size_t size, NativeWord* buffer)
{
    auto func = [&]<SIMD::UnrollingMode unrolling_mode> {
        SIMD::align_up(buffer, unrolling_mode);
        storage_mul_ntt_using<unrolling_mode>(data1, size1, data2, size2, result, size, buffer);
    };
    SIMD::use_last_supported_unrolling_mode_from<SIMD::NONE, SIMD::AVX2>(func);
}

// ISA specific code
#if SIMD_CAN_POSSIBLY_SUPPORT_AVX2
using SIMD::u64x4, SIMD::c8x32;

#    define TARGET_AVX2 __attribute__((target("avx2")))

static ALWAYS_INLINE TARGET_AVX2 u64x4 mod_reduce_vec_96(u64x4 low, u64x4 middle)
{
    middle -= middle >> 32;
    u64x4 result = low + middle;
    u64x4 corrected_result = result - modulus;
    u64x4 mask = result < middle || result >= modulus;
    return (u64x4)__builtin_ia32_pblendvb256((c8x32)result, (c8x32)corrected_result, (c8x32)mask);
}

static ALWAYS_INLINE TARGET_AVX2 u64x4 mod_reduce_vec_128(u64x4 low, u64x4 middle, u64x4 high)
{
    u64x4 low2 = low - high;

    u64x4 corrected_low2 = low2 + modulus;
    u64x4 mask = high > low;
    low2 = (u64x4)__builtin_ia32_pblendvb256((c8x32)low2, (c8x32)corrected_low2, (c8x32)mask);

    middle -= middle >> 32;
    u64x4 result = low2 + middle;

    u64x4 corrected_result = result - modulus;
    mask = result < middle || result >= modulus;
    return (u64x4)__builtin_ia32_pblendvb256((c8x32)result, (c8x32)corrected_result, (c8x32)mask);
}

static ALWAYS_INLINE TARGET_AVX2 u64x4 mod_reduce_vec_159(u64x4 middle, u64x4 high)
{
    middle -= middle >> 32;
    u64x4 corrected_result = middle - high;
    u64x4 result = corrected_result + modulus;
    u64x4 mask = result < middle || result >= modulus;
    return (u64x4)__builtin_ia32_pblendvb256((c8x32)result, (c8x32)corrected_result, (c8x32)mask);
}

template<size_t sc>
static ALWAYS_INLINE TARGET_AVX2 u64x4 mod_shift(u64x4 y, size_t shift)
{
    if constexpr (sc == 0) {
        return y;
    } else if constexpr (sc == 1) {
        return mod_reduce_vec_96(y << shift, y >> (64 - shift) << 32);
    } else if constexpr (sc == 2) {
        return mod_reduce_vec_128(y << shift, y >> (64 - shift) << 32, y >> (96 - shift));
    } else {
        return mod_reduce_vec_159(y << (shift - 32), y >> (96 - shift));
    }
}

static ALWAYS_INLINE TARGET_AVX2 void mod_add_sub(u64x4& xs, u64x4& ys)
{
    auto minus_ys = modulus - ys;
    auto added = xs + ys;
    auto corrected_add = added - modulus;
    u64x4 res_xs = (u64x4)__builtin_ia32_pblendvb256((c8x32)added, (c8x32)corrected_add, minus_ys <= xs);

    auto corrected_sub = xs + minus_ys;
    auto subtracted = corrected_sub - modulus;
    ys = (u64x4)__builtin_ia32_pblendvb256((c8x32)subtracted, (c8x32)corrected_sub, xs < ys);

    xs = res_xs;
}

template<>
ALWAYS_INLINE TARGET_AVX2 void ntt_convolve<SIMD::AVX2>(u64* a, size_t j, size_t part_len, size_t total_len, size_t shift)
{
    if (part_len <= 4) {
        ntt_convolve<SIMD::NONE>(a, j, part_len, total_len, shift);
        return;
    }
    shift_dispatch([&]<size_t sc>() TARGET_AVX2 {
        part_len /= 4;

        auto left = (u64x4*)(a + j);
        auto right = (u64x4*)(a + j + total_len);

        for (size_t h = 0; h < part_len; ++h, ++left, ++right) {
            auto xs = *left;
            auto ys = mod_shift<sc>(*right, shift);
            mod_add_sub(xs, ys);

            *left = xs;
            *right = ys;
        }
    },
        shift);
}

template<>
ALWAYS_INLINE TARGET_AVX2 void ntt_convolve2<SIMD::AVX2>(u64* a, size_t j, size_t n, size_t shift1, size_t shift2, size_t shift3)
{
    shift_dispatch([&]<size_t sc1, size_t sc2, size_t sc3>() TARGET_AVX2 {
        auto x_ptr = (u64x4*)(a + j);
        auto y_ptr = (u64x4*)(a + j + n);
        auto z_ptr = (u64x4*)(a + j + 2 * n);
        auto w_ptr = (u64x4*)(a + j + 3 * n);

        n /= 4;

        for (size_t i = 0; i < n; ++i) {
            auto x = *x_ptr, y = *y_ptr;
            auto z = mod_shift<sc1>(*z_ptr, shift1), w = mod_shift<sc1>(*w_ptr, shift1);
            mod_add_sub(x, z);
            mod_add_sub(y, w);

            y = mod_shift<sc2>(y, shift2);
            w = mod_shift<sc3>(w, shift3);
            mod_add_sub(x, y);
            mod_add_sub(z, w);

            *x_ptr++ = x;
            *y_ptr++ = y;
            *z_ptr++ = z;
            *w_ptr++ = w;
        }
    },
        shift1, shift2, shift3);
}

template<>
ALWAYS_INLINE TARGET_AVX2 void ntt_convolve3<SIMD::AVX2>(u64* a, size_t j, size_t part_len, size_t total_len, size_t shift1, size_t shift2, size_t shift3)
{
    if (part_len < 4)
        return ntt_convolve3<SIMD::NONE>(a, j, part_len, total_len, shift1, shift2, shift3);

    shift_dispatch([&]<size_t sc1, size_t sc2, size_t sc3>() TARGET_AVX2 {
        auto x_ptr = (u64x4*)(a + j);
        auto y_ptr = (u64x4*)(a + j + total_len);
        auto z_ptr = (u64x4*)(a + j + 2 * total_len);
        auto w_ptr = (u64x4*)(a + j + 3 * total_len);

        part_len /= 4;

        for (size_t i = 0; i < part_len; ++i) {
            auto x = *x_ptr, y = mod_shift<sc1>(*y_ptr, shift1), z = *z_ptr, w = mod_shift<sc1>(*w_ptr, shift1);
            mod_add_sub(x, y);
            mod_add_sub(z, w);

            z = mod_shift<sc2>(z, shift2);
            w = mod_shift<sc3>(w, shift3);
            mod_add_sub(x, z);
            mod_add_sub(y, w);

            *x_ptr++ = x;
            *y_ptr++ = y;
            *z_ptr++ = z;
            *w_ptr++ = w;
        }
    },
        shift1, shift2, shift3);
}

template<>
ALWAYS_INLINE TARGET_AVX2 void ntt_multiply_arbitrary<SIMD::AVX2>(size_t from, size_t to, size_t scale, u64* a, u64* local_factors)
{
    if (to - from < 4)
        return ntt_multiply_arbitrary<SIMD::NONE>(from, to, scale, a, local_factors);

    u64 scale_factor = local_factors[scale];
    u64 factor = 1;

    for (size_t i = from; i < to; i += 4) {
        auto x0 = static_cast<unsigned __int128>(a[i]) * factor;
        factor = mod_mul(factor, scale_factor);
        auto x1 = static_cast<unsigned __int128>(a[i + 1]) * factor;
        factor = mod_mul(factor, scale_factor);
        auto x2 = static_cast<unsigned __int128>(a[i + 2]) * factor;
        factor = mod_mul(factor, scale_factor);
        auto x3 = static_cast<unsigned __int128>(a[i + 3]) * factor;
        factor = mod_mul(factor, scale_factor);

        u64x4 low = { (u64)x0, (u64)x1, (u64)x2, (u64)x3 };
        u64x4 middle = { (u64)(x0 >> 64), (u64)(x1 >> 64), (u64)(x2 >> 64), (u64)(x3 >> 64) };
        u64x4 high = middle >> 32;
        middle <<= 32;
        *(u64x4*)(a + i) = mod_reduce_vec_128(low, middle, high);
    }
}

template<>
ALWAYS_INLINE TARGET_AVX2 void ntt_multiply_arbitrary2<SIMD::AVX2>(size_t from, size_t to, size_t scale, size_t idx_mask, u64* a, u64* local_factors)
{
    if (to - from <= 4)
        return ntt_multiply_arbitrary2<SIMD::NONE>(from, to, scale, idx_mask, a, local_factors);

    size_t factor_idx = 0;

    for (size_t i = from; i < to; i += 4) {
        auto x0 = static_cast<unsigned __int128>(a[i]) * local_factors[factor_idx];
        factor_idx += scale;
        factor_idx &= idx_mask;
        auto x1 = static_cast<unsigned __int128>(a[i + 1]) * local_factors[factor_idx];
        factor_idx += scale;
        factor_idx &= idx_mask;
        auto x2 = static_cast<unsigned __int128>(a[i + 2]) * local_factors[factor_idx];
        factor_idx += scale;
        factor_idx &= idx_mask;
        auto x3 = static_cast<unsigned __int128>(a[i + 3]) * local_factors[factor_idx];
        factor_idx += scale;
        factor_idx &= idx_mask;

        u64x4 low = { (u64)x0, (u64)x1, (u64)x2, (u64)x3 };
        u64x4 middle = { (u64)(x0 >> 64), (u64)(x1 >> 64), (u64)(x2 >> 64), (u64)(x3 >> 64) };
        u64x4 high = middle >> 32;
        middle <<= 32;
        *(u64x4*)(a + i) = mod_reduce_vec_128(low, middle, high);
    }
}

template<>
ALWAYS_INLINE TARGET_AVX2 void ntt_multiply_constant<SIMD::AVX2>(u64* a, size_t from, size_t to, u64 value)
{
    for (size_t i = from; i < to; i += 4) {
        auto x0 = static_cast<unsigned __int128>(a[i]) * value;
        auto x1 = static_cast<unsigned __int128>(a[i + 1]) * value;
        auto x2 = static_cast<unsigned __int128>(a[i + 2]) * value;
        auto x3 = static_cast<unsigned __int128>(a[i + 3]) * value;

        u64x4 low = { (u64)x0, (u64)x1, (u64)x2, (u64)x3 };
        u64x4 middle = { (u64)(x0 >> 64), (u64)(x1 >> 64), (u64)(x2 >> 64), (u64)(x3 >> 64) };
        u64x4 high = middle >> 32;
        middle <<= 32;
        *(u64x4*)(a + i) = mod_reduce_vec_128(low, middle, high);
    }
}

template TARGET_AVX2 void nttf<SIMD::AVX2>(size_t, size_t, size_t, u64*, u64*, u64*, u64*, NativeWord*);
template TARGET_AVX2 void nttr<SIMD::AVX2>(size_t, size_t, size_t, u64*, u64*, u64*, u64*);
#endif

}
