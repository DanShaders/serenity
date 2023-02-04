/*
 * Copyright (c) 2023, Dan Klishch <danilklishch@gmail.com>
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <AK/Diagnostics.h>
#include <AK/UFixedBigInt.h>

namespace AK {

template<size_t divident_bit_size, size_t divisor_bit_size, bool restore_remainder>
constexpr void div_mod_internal(
    StaticStorage<false, divident_bit_size> const& operand1,
    StaticStorage<false, divisor_bit_size> const& operand2,
    StaticStorage<false, divident_bit_size>& quotient,
    StaticStorage<false, divisor_bit_size>& remainder)
{
    size_t divident_len = operand1.size(), divisor_len = operand2.size();
    while (divisor_len > 0 && !operand2[divisor_len - 1])
        --divisor_len;
    while (divident_len > 0 && !operand1[divident_len - 1])
        --divident_len;

    // FIXME: Should raise SIGFPE instead
    VERIFY(divisor_len); // VERIFY(divisor != 0)

    // Fast paths
    if (divisor_len == 1 && operand2[0] == 1) { // divisor == 1
        quotient = operand1;
        if constexpr (restore_remainder)
            storage_set(0, remainder);
        return;
    }

    if (divident_len < divisor_len) { // divident < divisor
        storage_set(0, quotient);
        if constexpr (restore_remainder)
            remainder = operand1;
        return;
    }

    if (divisor_len == 1 && divident_len == 1) { // NativeWord / NativeWord
        storage_set(operand1[0] / operand2[0], quotient);
        if constexpr (restore_remainder)
            storage_set(operand1[0] % operand2[0], remainder);
        return;
    }

    if (divisor_len == 1) { // BigInt by NativeWord
        auto u = (static_cast<DoubleWord>(operand1[divident_len - 1]) << word_size) + operand1[divident_len - 2];
        auto divisor = operand2[0];

        auto top = u / divisor;
        quotient[divident_len - 1] = static_cast<NativeWord>(top >> word_size);
        quotient[divident_len - 2] = static_cast<NativeWord>(top);

        auto carry = static_cast<NativeWord>(u % divisor);
        for (size_t i = divident_len - 2; i--;)
            quotient[i] = div_mod_words(operand1[i], carry, divisor, carry);
        for (size_t i = divident_len; i < quotient.size(); ++i)
            quotient[i] = 0;
        if constexpr (restore_remainder)
            storage_set(carry, remainder);
        return;
    }

    // Knuth's algorithm D
    StaticStorage<false, divident_bit_size + word_size> divident;
    storage_copy(operand1, divident);
    auto divisor = operand2;

    // D1. Normalize
    // FIXME: Investigate GCC producing bogus -Warray-bounds when dividing u128 by u32. This code
    //        should not be reachable at all in this case because fast paths above cover all cases
    //        when `operand2.size() == 1`.
    AK_IGNORE_DIAGNOSTIC("-Warray-bounds", size_t shift = count_leading_zeroes(divisor[divisor_len - 1]);)
    storage_shift_left(divident, shift, divident);
    storage_shift_left(divisor, shift, divisor);

    auto divisor_approx = divisor[divisor_len - 1];

    for (size_t i = divident_len + 1; i-- > divisor_len;) {
        // D3. Calculate qhat
        NativeWord qhat;
        VERIFY(divident[i] <= divisor_approx);
        if (divident[i] == divisor_approx) {
            qhat = max_word;
        } else {
            NativeWord rhat;
            qhat = div_mod_words(divident[i - 1], divident[i], divisor_approx, rhat);

            auto is_qhat_too_large = [&] {
                return UFixedBigInt<word_size> { qhat }.wide_multiply(divisor[divisor_len - 2]) > u128 { divident[i - 2], rhat };
            };
            if (is_qhat_too_large()) {
                --qhat;
                bool carry;
                add_words(rhat, divisor_approx, rhat, carry);
                if (is_qhat_too_large())
                    --qhat;
            }
        }

        // D4. Multiply & subtract
        NativeWord mul_carry = 0;
        bool sub_carry = false;
        for (size_t j = 0; j < divisor_len; ++j) {
            auto mul_result = UFixedBigInt<word_size> { qhat }.wide_multiply(divisor[j]) + mul_carry;
            auto& output = divident[i + j - divisor_len];
            sub_words(output, mul_result.low(), output, sub_carry);
            mul_carry = mul_result.high();
        }
        sub_words(divident[i], mul_carry, divident[i], sub_carry);

        if (sub_carry) {
            // D6. Add back
            auto divident_part = UnsignedStorageSpan { divident.data() + i - divisor_len, divisor_len + 1 };
            VERIFY(storage_add<false>(divident_part, divisor, divident_part));
        }

        quotient[i - divisor_len] = qhat - sub_carry;
    }

    for (size_t i = divident_len - divisor_len + 1; i < quotient.size(); ++i)
        quotient[i] = 0;

    // D8. Unnormalize
    if constexpr (restore_remainder)
        storage_shift_right(UnsignedStorageSpan { divident.data(), divisor_len }, shift, remainder);
}
}
