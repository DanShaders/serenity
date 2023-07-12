/*
 * Copyright (c) 2023, Dan Klishch <danilklishch@gmail.com>
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <AK/BigIntBase.h>
#include <AK/SIMD.h>
#include <AK/UFixedBigInt.h>
#include <immintrin.h>

#define LIBMP_DEBUG_PROFILING

#ifdef LIBMP_DEBUG_PROFILING
#    include <AK/OwnPtr.h>
#    include <AK/ScopeGuard.h>
#    include <AK/String.h>
#    include <AK/Time.h>
#    include <AK/Vector.h>

namespace {
class Profiler {
public:
    Profiler(String name)
    {
        nodes = { {
            .parent = 0,
            .enter_time = MonotonicTime::now(),
            .leave_time = {},
            .name = name,
            .children = {},
        } };
    }

    void push_entry(String node_name)
    {
        auto now = MonotonicTime::now();

        nodes[ptr].children.append(nodes.size());
        nodes.append({ ptr, now, {}, move(node_name), {} });
        ptr = nodes.size() - 1;
    }

    void pop_entry()
    {
        nodes[ptr].leave_time = MonotonicTime::now();
        ptr = nodes[ptr].parent;
    }

    void stop()
    {
        nodes[0].leave_time = MonotonicTime::now();
        VERIFY(ptr == 0);
    }

    void print() const
    {
        print_internal(0, 0);
    }

    void clear()
    {
        nodes = { {} };
        ptr = 0;
    }

private:
    struct Node {
        size_t parent;
        Optional<MonotonicTime> enter_time, leave_time;
        String name;
        Vector<size_t> children;
    };

    Vector<Node> nodes;
    size_t ptr = 0;

    void print_internal(int i, int level) const
    {
        u64 total = (nodes[i].leave_time.value() - nodes[i].enter_time.value()).to_nanoseconds();
        u64 net = total;
        for (auto child : nodes[i].children)
            net -= (nodes[child].leave_time.value() - nodes[child].enter_time.value()).to_nanoseconds();

        StringBuilder indent;
        for (int i = 0; i < 3 * level; ++i)
            indent.append(' ');
        if (!nodes[i].children.is_empty())
            dbgln("{}{} ({:06}s, local {:06}s)", MUST(indent.to_string()), nodes[i].name, total / 1e9, net / 1e9);
        else
            dbgln("{}{} ({:06}s)", MUST(indent.to_string()), nodes[i].name, total / 1e9);

        for (auto child : nodes[i].children)
            print_internal(child, level + 1);
    }
};

OwnPtr<Profiler> global_profiler;
}

#    define CONCAT_IMPL(x, y) x##y
#    define CONCAT(x, y) CONCAT_IMPL(x, y)

#    define PROFILER_PUSH_ENTRY(format, ...) global_profiler->push_entry(MUST(String::formatted(format __VA_OPT__(, ) __VA_ARGS__)))
#    define PROFILER_POP_ENTRY() global_profiler->pop_entry()
#    define PROFILER_SCOPE(format, ...)           \
        PROFILER_PUSH_ENTRY(format, __VA_ARGS__); \
        auto CONCAT(profiler_guard_, __COUNTER__) = ScopeGuard([] { PROFILER_POP_ENTRY(); })
#    define PROFILER_NEW_RUN(format, ...) \
        global_profiler = OwnPtr<Profiler>::lift(new Profiler(MUST(String::formatted(format __VA_OPT__(, ) __VA_ARGS__))));
#    define PROFILER_STOP() global_profiler->stop()
#    define PROFILER_PRINT() global_profiler->print()
#else
#    define PROFILER_PUSH_ENTRY(format, ...)
#    define PROFILER_POP_ENTRY()
#    define PROFILER_SCOPE(format, ...)
#    define PROFILER_NEW_RUN(format, ...)
#    define PROFILER_STOP()
#    define PROFILER_PRINT()
#endif

// TODO: update description

// ===== Multiplication =====
// Time complexity: O(n log n)
//
// Very high-level description of NTT for this particular modulus can be found at
// https://cp4space.hatsya.com/2021/09/01/an-efficient-prime-for-number-theoretic-transforms/ .
//
// On the algorithmic level, we do discrete Fourier transform over Z/modulus using mixed-radix
// Cooley-Tukey FFT. Radix (`1 << inner_iters`) is 64 (as opposed to 2 in naive implementations)
// except for the second iteration of the loops in forward and backward NTTs (when it can be smaller
// in order to match `n`).
//
// Multiplications by arbitrary twiddle factors (happening between some of the outer loops
// iterations) are done in `ntt_multiply_arbitrary`, `ntt_multiply_arbitrary2` and
// `ntt_multiply_constant`. It should be noted that we gather all of the required twiddle factors
// from twiddle_factors into local_factors/twiddle_buffer to improve cache locality during
// multiplication passes (except in ntt_multiply_constant, for obvious reasons).
//
// Twiddle factors of inner NTTs (of size 64 or less) happen to be the powers of 2. "Canonical"
// butterfly transform is located in `ntt_convolve`. In the presence of SIMD unrolling, it is
// beneficial to fuse convolution passes for the adjacent `log_len`s. This is done in
// `ntt_convolve2` (for forward direction) and `ntt_convolve3` (for backward direction).
// Unfortunately, compilers are still not smart enough to fuse these loops automatically (even if
// leave the contents of `ntt_convolve\d` to be invocations of corresponding `ntt_convolve`).

namespace AK::Detail {
namespace {
// clang-format off
constexpr u32 reversed_bits_64[] = {
    0, 32, 16, 48, 8,  40, 24, 56, 4, 36, 20, 52, 12, 44, 28, 60,
    2, 34, 18, 50, 10, 42, 26, 58, 6, 38, 22, 54, 14, 46, 30, 62,
    1, 33, 17, 49, 9,  41, 25, 57, 5, 37, 21, 53, 13, 45, 29, 61,
    3, 35, 19, 51, 11, 43, 27, 59, 7, 39, 23, 55, 15, 47, 31, 63,
};
// clang-format on

u32 reverse_up_to_6_bits(u32 value, u32 bits)
{
    return reversed_bits_64[value] >> (6 - bits);
}

constexpr u64 modulus = 0xffff'ffff'0000'0001ULL;

// (low + (middle << 64) + (high << 96)) % modulus
//
// NOTE: This is the only code from the article. I do not know how to write the following 10 lines
//       in a different way, so the function is directly copied.
u64 mod_reduce(u64 low, u64 middle, u64 high)
{
    // VERIFY(middle < 1ULL << 32);

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

// (a * b) % modulus
u64 mod_mul(u64 a, u64 b)
{
    u128 c = AK::UFixedBigInt<64>(a).wide_multiply(b);
    return mod_reduce(c.low(), c.high() & 0xffff'ffffULL, c.high() >> 32);
}

// (a + b) % modulus
u64 mod_add(u64 a, u64 b)
{
    return a + b - (b < modulus - a ? 0 : modulus);
}

// (a - b) % modulus
u64 mod_sub(u64 a, u64 b)
{
    return a - b + (a < b ? modulus : 0);
}

// a_new = (a + b) % modulus, b_new = (a - b) % modulus
void mod_add_sub(u64& a, u64& b)
{
    u64 add = a + b - (b < modulus - a ? 0 : modulus);
    u64 sub = a - b + (a < b ? modulus : 0);
    a = add;
    b = sub;
}

// Throughout FFT, we need to shift integers to the left by up to 95 bits and reduce them modulo
// `modulus`. The algorithm of shifting and reducing itself is slightly different in these 4 cases.
// Furthermore, we usually need to shift not one, but multiple integers by the same number of bits.
// So, to achieve maximum efficiency, we generate copies of inner loops for each type of the shift.
enum class ShiftType {
    IDENTITY, // shift == 0
    LQ_32,    // 0 < shift <= 32
    LE_64,    // 32 < shift < 64
    LE_96,    // 64 <= shift < 96
};

// This allows us to call functions, templated by ShiftType, cleanly.
template<typename Func, ShiftType... shift_types>
ALWAYS_INLINE auto shift_dispatch(Func func)
{
    return func.template operator()<shift_types...>();
}

template<typename Func, ShiftType... shift_types>
ALWAYS_INLINE auto shift_dispatch(Func func, u32 current_shift, SameAs<u32> auto... remaining_shifts)
{
    if (current_shift == 0) {
        return shift_dispatch<Func, shift_types..., ShiftType::IDENTITY>(func, remaining_shifts...);
    } else if (current_shift <= 32) {
        return shift_dispatch<Func, shift_types..., ShiftType::LQ_32>(func, remaining_shifts...);
    } else if (current_shift < 64) {
        return shift_dispatch<Func, shift_types..., ShiftType::LE_64>(func, remaining_shifts...);
    } else {
        return shift_dispatch<Func, shift_types..., ShiftType::LE_96>(func, remaining_shifts...);
    }
}

// (a << shift) % modulus
//
// NOTE: We rely on the compiler to optimize away unnecessary checks in `mod_reduce` in the presence
//       of the constant parameters.
template<ShiftType shift_type>
u64 mod_shift(u64 a, u32 shift)
{
    // VERIFY(shift < 96);

    switch (shift_type) {
    case ShiftType::IDENTITY:
        return a;

    case ShiftType::LQ_32:
        return mod_reduce(a << shift, a >> (64 - shift), 0);

    case ShiftType::LE_64:
        return mod_reduce(a << shift, static_cast<u32>(a >> (64 - shift)), a >> (96 - shift));

    case ShiftType::LE_96:
        return mod_reduce(0, a << (shift - 64), a >> (96 - shift));
    }
}

// The classical FFT's butterfly transform
template<SIMD::UnrollingMode>
ALWAYS_INLINE void ntt_convolve(u64* a, size_t j, size_t part_len, size_t total_len, size_t shift)
{
    shift_dispatch([&]<ShiftType shift_type> {
        for (size_t h = j; h < j + part_len; ++h) {
            auto x = a[h], y = mod_shift<shift_type>(a[h + total_len], shift);
            a[h] = mod_add(x, y), a[h + total_len] = mod_sub(x, y);
        }
    },
        shift);
}

// Fused two iterations of NTT in the forward direction, equivalent to the following three calls:
//   ntt_convolve<unrolling_mode>(a, j, 2 * part_len, 2 * part_len, shift1);
//   ntt_convolve<unrolling_mode>(a, j, part_len, part_len, shift2);
//   ntt_convolve<unrolling_mode>(a, j + 2 * part_len, part_len, part_len, shift3);
template<SIMD::UnrollingMode>
void ntt_convolve2(u64* a, size_t j, size_t part_len, u32 shift1, u32 shift2, u32 shift3)
{
    shift_dispatch([&]<ShiftType s1, ShiftType s2, ShiftType s3> {
        for (u64 h = 0; h < part_len; ++h) {
            auto x = a[j + h];
            auto y = a[j + h + part_len];
            auto z = a[j + h + 2 * part_len];
            auto w = a[j + h + 3 * part_len];
            z = mod_shift<s1>(z, shift1);
            w = mod_shift<s1>(w, shift1);
            mod_add_sub(x, z);
            mod_add_sub(y, w);
            y = mod_shift<s2>(y, shift2);
            w = mod_shift<s3>(w, shift3);
            mod_add_sub(x, y);
            mod_add_sub(z, w);

            a[j + h] = x;
            a[j + h + part_len] = y;
            a[j + h + 2 * part_len] = z;
            a[j + h + 3 * part_len] = w;
        }
    },
        shift1, shift2, shift3);
}

// Fused two iterations of NTT in the backward direction, equivalent to the following four calls:
//   ntt_convolve<unrolling_mode>(a, j, part_len, total_len, shift1);
//   ntt_convolve<unrolling_mode>(a, j + 2 * total_len, part_len, total_len, shift1);
//   ntt_convolve<unrolling_mode>(a, j, part_len, 2 * total_len, shift2);
//   ntt_convolve<unrolling_mode>(a, j + total_len, part_len, 2 * total_len, shift3);
template<SIMD::UnrollingMode>
void ntt_convolve3(u64* a, size_t j, size_t part_len, size_t total_len, u32 shift1, u32 shift2, u32 shift3)
{
    shift_dispatch([&]<ShiftType s1, ShiftType s2, ShiftType s3> {
        for (u64 h = 0; h < part_len; ++h) {
            auto x = a[j + h];
            auto y = a[j + h + total_len];
            auto z = a[j + h + 2 * total_len];
            auto w = a[j + h + 3 * total_len];
            y = mod_shift<s1>(y, shift1);
            w = mod_shift<s1>(w, shift1);
            mod_add_sub(x, y);
            mod_add_sub(z, w);
            z = mod_shift<s2>(z, shift2);
            w = mod_shift<s3>(w, shift3);
            mod_add_sub(x, z);
            mod_add_sub(y, w);

            a[j + h] = x;
            a[j + h + total_len] = y;
            a[j + h + 2 * total_len] = z;
            a[j + h + 3 * total_len] = w;
        }
    },
        shift1, shift2, shift3);
}

template<SIMD::UnrollingMode, int interleaving>
void ntt_multiply_arbitrary(size_t from, size_t to, size_t scale, u64* a, u64* local_factors)
{
    size_t scaling = local_factors[scale];
    size_t factor = 1;
    for (size_t i = from; i < to; ++i) {
        for (size_t copy = 0; copy < interleaving; ++copy) {
            u64 index = interleaving * i + copy;
            a[index] = mod_mul(a[index], factor);
        }
        factor = mod_mul(factor, scaling);
    }
}

template<SIMD::UnrollingMode>
void ntt_multiply_arbitrary2(size_t from, size_t to, size_t scale, size_t idx_mask, u64* a, u64* local_factors)
{
    size_t factor_idx = 0;

    for (size_t i = from; i < to; ++i) {
        a[i] = mod_mul(a[i], local_factors[factor_idx]);
        factor_idx += scale;
        factor_idx &= idx_mask;
    }
}

template<SIMD::UnrollingMode>
void ntt_multiply_constant(u64* a, size_t from, size_t to, u64 value)
{
    for (size_t i = from; i < to; ++i)
        a[i] = mod_mul(a[i], value);
}

struct PlanItem {
    u32 from;
    u32 to;
    u32 log_len;
    bool fused;
} plan[64];

struct PlanOptions {
    bool is_forward;
    bool fuse_if_possible;
};

void make_plan(u32& i, u32 l, u32 r, u32 log_len, PlanOptions const& options)
{
    if (r - l == 1)
        return;

    bool will_fuse = options.fuse_if_possible && (r - l > 2);

    u32 part1 = (3 * l + r) / 4;
    u32 mid = (l + r) / 2;
    u32 part3 = (l + 3 * r) / 4;

    if (options.is_forward) {
        plan[i++] = { l, r, log_len, will_fuse };
    }

    if (will_fuse) {
        make_plan(i, l, part1, log_len + 2, options);
        make_plan(i, part1, mid, log_len + 2, options);
        make_plan(i, mid, part3, log_len + 2, options);
        make_plan(i, part3, r, log_len + 2, options);
    } else {
        make_plan(i, l, mid, log_len + 1, options);
        make_plan(i, mid, r, log_len + 1, options);
    }

    if (!options.is_forward) {
        plan[i++] = { l, r, log_len, will_fuse };
    }
}

constexpr size_t one_sz = 1;

constexpr u32 reversed_idx_shift = 6;
constexpr u32 twiddle_sparse_shift = 6;
constexpr u32 nttm_iterations = 6;

template<SIMD::UnrollingMode unrolling_mode, int interleaving>
void nttf(size_t k, size_t n, u64* a, u64* twiddle_start, u64* twiddle_sparse, u64* twiddle_buffer, NativeWord* reversed_idx)
{
    PROFILER_SCOPE("nntf");

    for (size_t outer_log_len = 0; outer_log_len < k;) {
        PROFILER_SCOPE("outer_log_len {}/{}", outer_log_len, k);

        size_t inner_iters = outer_log_len == 0 ? 6 : (outer_log_len == 6 && k % 6 != 0 ? k % 6 : 6);

        // Unlike in `nttr`, I don't know what we've computed at this point. :)
        // We should have done something with blocks of `part_len` numbers in a[reversed_idx[...]].

        // Multiply by twiddle factors
        if (outer_log_len) {
            PROFILER_SCOPE("mult");

            size_t part_len = one_sz << outer_log_len;
            size_t parts = one_sz << inner_iters;
            size_t shift = k - inner_iters - outer_log_len;
            size_t total_len = one_sz << (k - outer_log_len);
            size_t inner_len = one_sz << shift;

            if (inner_len == 1) {
                VERIFY(inner_iters == nttm_iterations);
                VERIFY(inner_iters >= reversed_idx_shift);

                for (size_t i = 0; i < n; i += parts) {
                    size_t offset_in_part = reversed_idx[i >> inner_iters] >> inner_iters;
                    ntt_multiply_arbitrary<unrolling_mode, interleaving>(i, i + parts, offset_in_part, a, twiddle_start);
                }
            } else {
                VERIFY(shift >= twiddle_sparse_shift);

                u64* local_factors = twiddle_sparse;
                if (shift > twiddle_sparse_shift) {
                    local_factors = twiddle_buffer;
                    for (size_t i = 0; i < (n >> shift); ++i)
                        local_factors[i] = twiddle_sparse[i << (shift - twiddle_sparse_shift)];
                }

                for (size_t i = 0; i < n; i += total_len) {
                    size_t offset_in_part = reversed_idx[i >> (inner_iters + reversed_idx_shift)] >> (inner_iters + reversed_idx_shift) & (part_len - 1);
                    if (!offset_in_part)
                        continue;

                    for (size_t j = i, idx = 0; j < i + total_len; j += inner_len, (idx += offset_in_part) &= (n >> shift) - 1)
                        if (idx)
                            ntt_multiply_constant<unrolling_mode>(a, interleaving * j, interleaving * (j + inner_len), local_factors[idx]);
                }
            }
        }

        if (outer_log_len + inner_iters == k) {
            VERIFY(inner_iters == nttm_iterations);
            break;
        }

        // Do NTTs of size `parts`
        PROFILER_SCOPE("convolve");

        u32 plan_size = 0;
        bool should_fuse = unrolling_mode != SIMD::NONE && outer_log_len <= 6; // TODO
        make_plan(plan_size, 0, 1 << inner_iters, 0, { .is_forward = true, .fuse_if_possible = should_fuse });

        u64 part_len = one_sz << (k - outer_log_len - inner_iters);
        u64 block_len = part_len << inner_iters;
        for (u64 block = 0; block < n; block += block_len) {
            for (u32 plan_item = 0; plan_item < plan_size; ++plan_item) {
                auto [l, r, log_len, fused] = plan[plan_item];

                u64 current_offset = interleaving * (block + l * part_len);
                u64 current_len = interleaving * (r - l) / 2 * part_len;

                u32 shift1 = 3 * (reverse_up_to_6_bits(l, 6) << (inner_iters - log_len - 1));
                if (!fused) {
                    ntt_convolve<unrolling_mode>(a, current_offset, current_len, current_len, shift1);
                } else {
                    u32 shift2 = 3 * (reverse_up_to_6_bits(l, 6) << (inner_iters - log_len - 2));
                    u32 shift3 = 3 * (reverse_up_to_6_bits((l + r) / 2, 6) << (inner_iters - log_len - 2));
                    ntt_convolve2<unrolling_mode>(a, current_offset, current_len / 2, shift1, shift2, shift3);
                }
            }
        }
        outer_log_len += inner_iters;
    }
}

template<SIMD::UnrollingMode unrolling_mode>
void nttm(size_t n, u64* a)
{
    PROFILER_SCOPE("nttm");

    constexpr u64 reorder_bucket_size = 64;
    constexpr u64 reorder_buckets = 64;
    constexpr u64 reorder_buffer_size = reorder_bucket_size * reorder_buckets;

    VERIFY(n % reorder_buffer_size == 0);

    AK_CACHE_ALIGNED u64 reorder_buffer[2 * reorder_buffer_size];

    SIMD::u64x4 mask_load = { 0, 1, 2 * reorder_buckets, 2 * reorder_buckets + 1 };
    SIMD::u64x4 mask_store = { 0, reorder_bucket_size, 2 * reorder_bucket_size, 3 * reorder_bucket_size };

    for (u64 reorder_block = 0; reorder_block < n; reorder_block += reorder_buffer_size) {
        for (u64 i = 0; i < reorder_bucket_size; i += unrolling_mode == SIMD::NONE ? 1 : 2) {
            for (u64 bucket = 0; bucket < reorder_buckets; ++bucket) {
                if constexpr (unrolling_mode == SIMD::NONE) {
                    reorder_buffer[2 * (bucket * reorder_bucket_size + i)] = a[2 * (reorder_block + bucket + i * reorder_buckets)];
                    reorder_buffer[2 * (bucket * reorder_bucket_size + i) + 1] = a[2 * (reorder_block + bucket + i * reorder_buckets) + 1];
                } else {
                    *((__m256i*)(reorder_buffer + bucket * reorder_bucket_size * 2 + i * 2)) = _mm256_i64gather_epi64((long long*)(a + 2 * (reorder_block + bucket + i * reorder_buckets)), (__m256i)mask_load, 8);
                }
            }
        }

        for (size_t log_len = 0; log_len < 6; ++log_len) {
            size_t total_len = one_sz << (6 - log_len - 1);

            for (size_t i = 0; i < reorder_buckets; i += 2 * total_len) {
                size_t shift = 3 * (reversed_bits_64[i & 63] << (6 - log_len - 1) & 63);
                ntt_convolve<unrolling_mode>(reorder_buffer, 2 * i * reorder_bucket_size, 2 * total_len * reorder_bucket_size, 2 * total_len * reorder_bucket_size, shift);
            }
        }

        for (u64 i = 0; i < reorder_buffer_size; ++i) {
            reorder_buffer[i] = mod_mul(reorder_buffer[2 * i], reorder_buffer[2 * i + 1]);
        }

        for (u32 log_len = 0; log_len < 6; ++log_len) {
            u64 len = one_sz << log_len;
            u64 total_len = len;

            for (size_t i = 0; i < reorder_buckets; i += 2 * total_len) {
                for (size_t j = i; j < i + total_len; ++j) {
                    size_t shift = 3 * ((j - i) << (6 - log_len - 1) & 63);
                    ntt_convolve<unrolling_mode>(reorder_buffer, j * reorder_bucket_size, reorder_bucket_size, total_len * reorder_bucket_size, shift);
                }
            }
        }

        for (u64 i = 0; i < reorder_bucket_size; ++i) {
            for (u64 bucket = 0; bucket < reorder_buckets; bucket += unrolling_mode == SIMD::NONE ? 1 : 4) {
                if constexpr (unrolling_mode == SIMD::NONE) {
                    a[reorder_block + bucket + i * reorder_buckets] = reorder_buffer[bucket * reorder_bucket_size + i];
                } else {
                    *(__m256i*)(a + reorder_block + bucket + i * reorder_buckets) = _mm256_i64gather_epi64((long long*)(reorder_buffer + bucket * reorder_bucket_size + i), (__m256i)mask_store, 8);
                }
            }
        }
    }
}

template<SIMD::UnrollingMode unrolling_mode>
void nttr(size_t k, size_t n, u64* a, u64* twiddle_start, u64* twiddle_sparse, u64* twiddle_buffer)
{
    PROFILER_SCOPE("nttr");

    for (size_t outer_log_len = 0; outer_log_len < k;) {
        PROFILER_SCOPE("outer_log_len {}/{}", outer_log_len, k);

        size_t inner_iters = outer_log_len == 0 ? 6 : (outer_log_len == 6 && k % 6 != 0 ? k % 6 : 6);

        size_t part_len = one_sz << outer_log_len;
        size_t shift = k - inner_iters - outer_log_len;

        // We've computed NTT of each consecutive block of `part_len` numbers.
        // Now we want to merge `parts` consecutive NTTs into one NTT of size `block_len`.

        if (outer_log_len == 0) {
            VERIFY(inner_iters == nttm_iterations);
            outer_log_len += inner_iters;
            continue;
        }

        // Do NTTs of size `parts`
        u64* local_factors = twiddle_sparse;
        if (outer_log_len != 0) {
            VERIFY(shift == 0 || shift >= twiddle_sparse_shift);

            if (shift > twiddle_sparse_shift) {
                local_factors = twiddle_buffer;
                for (size_t i = 0; i < (n >> shift); ++i)
                    local_factors[i] = twiddle_sparse[i << (shift - twiddle_sparse_shift)];
            }
        }

        u32 plan_size = 0;
        bool should_fuse = outer_log_len != 6 && unrolling_mode != SIMD::NONE; // TODO
        make_plan(plan_size, 0, 1 << inner_iters, 0, { .is_forward = false, .fuse_if_possible = should_fuse });

        u64 block_len = part_len << inner_iters;
        for (u64 block = 0; block < n; block += block_len) {
            // Multiply
            if (outer_log_len) {
                for (size_t part_offset = 0; part_offset < block_len; part_offset += part_len) {
                    size_t part_in_block = reverse_up_to_6_bits(part_offset >> outer_log_len, inner_iters);
                    if (!part_in_block)
                        continue;

                    if (shift) {
                        // Since twiddle_buffer is usually small enough to fit in the cache, it's faster
                        // to scatter-gather values from it rather than recompute them.
                        ntt_multiply_arbitrary2<unrolling_mode>(block + part_offset, block + part_offset + part_len, part_in_block, (n >> shift) - 1, a, local_factors);
                    } else {
                        ntt_multiply_arbitrary<unrolling_mode, 1>(part_offset, part_offset + part_len, part_in_block, a, twiddle_start);
                    }
                }
            }

            for (u32 plan_item = 0; plan_item < plan_size; ++plan_item) {
                auto [l, r, log_len, fused] = plan[plan_item];
                u64 base = block + l * part_len;

                u64 current_len = one_sz << (outer_log_len + inner_iters - 1 - log_len);

                if (!fused) {
                    for (u64 part = 0; part < current_len; part += part_len) {
                        size_t shift = 3 * (part >> outer_log_len << (6 - inner_iters + log_len));
                        ntt_convolve<unrolling_mode>(a, base + part, part_len, current_len, shift);
                    }
                } else {
                    current_len /= 2;

                    for (size_t part = 0; part < current_len; part += part_len) {
                        u32 shift1 = 3 * (part >> outer_log_len << (7 - inner_iters + log_len));
                        u32 shift2 = 3 * (part >> outer_log_len << (6 - inner_iters + log_len));
                        u32 shift3 = 3 * ((part + current_len) >> outer_log_len << (6 - inner_iters + log_len));
                        ntt_convolve3<unrolling_mode>(a, base + part, part_len, current_len, shift1, shift2, shift3);
                    }
                }
            }
        }
        outer_log_len += inner_iters;
    }
}

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

template<SIMD::UnrollingMode unrolling_mode>
void storage_mul_ntt_using(NativeWord const* data1, size_t size1, NativeWord const* data2, size_t size2, NativeWord* result, size_t size, NativeWord* buffer)
{
    PROFILER_NEW_RUN("storage_mul_ntt");

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
    VERIFY(k >= 12);

    // Count twiddle factors
    PROFILER_PUSH_ENTRY("twiddle factors");
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
    PROFILER_POP_ENTRY();

    SIMD::align_up(buffer, unrolling_mode);

    // Split arguments into u16 words
    PROFILER_PUSH_ENTRY("split");

    u64* operands = reinterpret_cast<u64*>(buffer);
    buffer += (word_size == 32 ? 2 : 1) * 2 * n; // u64[2 * n]

    auto split_at = [&](u64 base, NativeWord word) {
        if constexpr (word_size == 32) {
            operands[base] = word & 0xffff;
            operands[base + 2] = word >> 16;
        } else {
            operands[base] = static_cast<u16>(word);
            operands[base + 2] = static_cast<u16>(word >> 16);
            operands[base + 4] = static_cast<u16>(word >> 32);
            operands[base + 6] = static_cast<u16>(word >> 48);
        }
    };

    u64 base = 0;
    for (u64 i = 0; i < min(size1, size2); ++i, base += word_size == 32 ? 4 : 8) {
        split_at(base, data1[i]);
        split_at(base + 1, data2[i]);
    }
    for (u64 i = min(size1, size2); i < max(size1, size2); ++i, base += word_size == 32 ? 4 : 8) {
        split_at(base, i < size1 ? data1[i] : 0);
        split_at(base + 1, i < size2 ? data2[i] : 0);
    }
    memset(operands + base, 0, sizeof(NativeWord) * (buffer - operands - base));
    PROFILER_POP_ENTRY();

    // Count reversed bit representations of indexes
    PROFILER_PUSH_ENTRY("reversed_idx");
    NativeWord* reversed_idx = allocate_native(n >> 6);
    if (n >= 64) {
        reversed_idx[0] = 0;
        for (size_t i = 0; i < k - 6; ++i)
            reversed_idx[one_sz << i] = one_sz << (k - i - 1);
        for (size_t i = 1; i < (n >> 6); ++i)
            reversed_idx[i] = reversed_idx[i & (i - 1)] ^ reversed_idx[i ^ (i & (i - 1))];
    }
    PROFILER_POP_ENTRY();

    // "Allocate" twiddle factors buffer
    u64* twiddle_buffer = reinterpret_cast<u64*>(buffer);

    // Multiplication time has come.
    nttf<unrolling_mode, 2>(k, n, operands, twiddle_start, twiddle_sparse, twiddle_buffer, reversed_idx);
    nttm<unrolling_mode>(n, operands);
    nttr<unrolling_mode>(k, n, operands, twiddle_start, twiddle_sparse, twiddle_buffer);

    PROFILER_PUSH_ENTRY("return");
    u64 modular_inverse_of_n = modulus - (modulus >> k); // (modular_inverse_of_n * n) % modulus == 1

    // Copy everything they want from us to the result
    constexpr size_t scale = word_size / 16;
    memset(result, 0, sizeof(NativeWord) * size);

    u64 carry = 0;
    for (size_t i = 0; i < min(n, size * scale); ++i) {
        bool local_carry = false;

        auto current_word = mod_mul(operands[(n - i) & (n - 1)], modular_inverse_of_n);

        if constexpr (word_size == 64) {
            carry = add_words(current_word, carry, local_carry);
        } else {
            NativeWord low_carry, high_carry;
            low_carry = add_words(carry, current_word, local_carry);
            high_carry = add_words(carry >> 32, current_word >> 32, local_carry);
            carry = (static_cast<u64>(high_carry) << 32) + low_carry;
        }

        result[i / scale] |= (carry & 0xffff) << (i % scale * 16);
        carry >>= 16;
        carry += static_cast<u64>(local_carry) << 48;
    }
    PROFILER_POP_ENTRY();

    PROFILER_STOP();
    PROFILER_PRINT();
}

// ISA specific code
#if SIMD_CAN_POSSIBLY_SUPPORT_AVX2
using SIMD::u64x4, SIMD::c8x32;

#    define TARGET_AVX2 __attribute__((target("avx2")))

TARGET_AVX2 u64x4 mod_reduce_vec_128(u64x4 low, u64x4 middle, u64x4 high)
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

template TARGET_AVX2 void ntt_convolve<SIMD::AVX2>(u64* a, size_t j, size_t part_len, size_t total_len, size_t shift);
template TARGET_AVX2 void ntt_convolve2<SIMD::AVX2>(u64* a, size_t j, size_t n, u32 shift1, u32 shift2, u32 shift3);
template TARGET_AVX2 void ntt_convolve3<SIMD::AVX2>(u64* a, size_t j, size_t part_len, size_t total_len, u32 shift1, u32 shift2, u32 shift3);

// template TARGET_AVX2 void ntt_multiply_arbitrary<SIMD::AVX2, 1>(size_t from, size_t to, size_t scale, u64* a, u64* local_factors);

template<>
TARGET_AVX2 void ntt_multiply_arbitrary<SIMD::AVX2, 1>(size_t from, size_t to, size_t scale, u64* a, u64* local_factors)
{
    if (to - from < 4)
        return ntt_multiply_arbitrary<SIMD::NONE, 1>(from, to, scale, a, local_factors);

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

// template TARGET_AVX2 void ntt_multiply_arbitrary<SIMD::AVX2, 2>(size_t from, size_t to, size_t scale, u64* a, u64* local_factors);

template<>
TARGET_AVX2 void ntt_multiply_arbitrary<SIMD::AVX2, 2>(size_t from, size_t to, size_t scale, u64* a, u64* local_factors)
{
    if (to - from < 4)
        return ntt_multiply_arbitrary<SIMD::NONE, 2>(from, to, scale, a, local_factors);

    u64 scale_factor = local_factors[scale];
    u64 factor = 1;

    for (size_t i = 2 * from; i < 2 * to; i += 4) {
        auto x0 = static_cast<unsigned __int128>(a[i]) * factor;
        auto x1 = static_cast<unsigned __int128>(a[i + 1]) * factor;
        factor = mod_mul(factor, scale_factor);
        auto x2 = static_cast<unsigned __int128>(a[i + 2]) * factor;
        auto x3 = static_cast<unsigned __int128>(a[i + 3]) * factor;
        factor = mod_mul(factor, scale_factor);

        u64x4 low = { (u64)x0, (u64)x1, (u64)x2, (u64)x3 };
        u64x4 middle = { (u64)(x0 >> 64), (u64)(x1 >> 64), (u64)(x2 >> 64), (u64)(x3 >> 64) };
        u64x4 high = middle >> 32;
        middle <<= 32;
        *(u64x4*)(a + i) = mod_reduce_vec_128(low, middle, high);
    }
}

// template TARGET_AVX2 void ntt_multiply_arbitrary2<SIMD::AVX2>(size_t from, size_t to, size_t scale, size_t idx_mask, u64* a, u64* local_factors);

template<>
TARGET_AVX2 void ntt_multiply_arbitrary2<SIMD::AVX2>(size_t from, size_t to, size_t scale, size_t idx_mask, u64* a, u64* local_factors)
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

        AK::taint_for_optimizer(x0);
        AK::taint_for_optimizer(x1);
        AK::taint_for_optimizer(x2);
        AK::taint_for_optimizer(x3);

        u64x4 low = { (u64)x0, (u64)x1, (u64)x2, (u64)x3 };
        u64x4 middle = { (u64)(x0 >> 64), (u64)(x1 >> 64), (u64)(x2 >> 64), (u64)(x3 >> 64) };
        u64x4 high = middle >> 32;
        middle <<= 32;
        *(u64x4*)(a + i) = mod_reduce_vec_128(low, middle, high);
    }
}

// template TARGET_AVX2 void ntt_multiply_constant<SIMD::AVX2>(u64* a, size_t from, size_t to, u64 value);

template<>
TARGET_AVX2 void ntt_multiply_constant<SIMD::AVX2>(u64* a, size_t from, size_t to, u64 value)
{
    for (size_t i = from; i < to; i += 4) {
        auto x0 = static_cast<unsigned __int128>(a[i]) * value;
        auto x1 = static_cast<unsigned __int128>(a[i + 1]) * value;
        auto x2 = static_cast<unsigned __int128>(a[i + 2]) * value;
        auto x3 = static_cast<unsigned __int128>(a[i + 3]) * value;

        AK::taint_for_optimizer(x0);
        AK::taint_for_optimizer(x1);
        AK::taint_for_optimizer(x2);
        AK::taint_for_optimizer(x3);

        u64x4 low = { (u64)x0, (u64)x1, (u64)x2, (u64)x3 };
        u64x4 middle = { (u64)(x0 >> 64), (u64)(x1 >> 64), (u64)(x2 >> 64), (u64)(x3 >> 64) };
        u64x4 high = middle >> 32;
        middle <<= 32;
        *(u64x4*)(a + i) = mod_reduce_vec_128(low, middle, high);
    }
}

template TARGET_AVX2 void nttf<SIMD::AVX2, 2>(size_t, size_t, u64*, u64*, u64*, u64*, NativeWord*);
template TARGET_AVX2 void nttr<SIMD::AVX2>(size_t, size_t, u64*, u64*, u64*, u64*);
template TARGET_AVX2 void nttm<SIMD::AVX2>(size_t, u64*);
#endif
}

void storage_mul_smart(NativeWord const* data1, size_t size1, NativeWord const* data2, size_t size2, NativeWord* result, size_t size, NativeWord* buffer)
{
    auto func = [&]<SIMD::UnrollingMode unrolling_mode> {
        SIMD::align_up(buffer, unrolling_mode);
        storage_mul_ntt_using<unrolling_mode>(data1, size1, data2, size2, result, size, buffer);
    };
    SIMD::use_last_supported_unrolling_mode_from<SIMD::NONE, SIMD::AVX2>(func);
}

}
