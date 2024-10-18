/*
 * Copyright (c) 2024, Dan Klishch <danilklishch@gmail.com>
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <AK/AsyncStream.h>
#include <LibCore/Event.h>
#include <LibCore/ThreadEventQueue.h>
#include <LibTest/Macros.h>

namespace Test {

enum class StreamCloseExpectation {
    Reset,
    Close,
};

class AsyncMemoryInputStream final : public AsyncInputStream {
public:
    AsyncMemoryInputStream(StringView data, StreamCloseExpectation expectation, Vector<size_t>&& chunks);
    ~AsyncMemoryInputStream();

    virtual void cancel() override;
    virtual Coroutine<void> reset() override;
    virtual Coroutine<ErrorOr<void>> close() override;
    virtual bool is_open() const override;

    virtual Coroutine<ErrorOr<bool>> enqueue_some(Badge<AsyncInputStream>) override;
    virtual ReadonlyBytes buffered_data_unchecked(Badge<AsyncInputStream>) const override;
    virtual void dequeue(Badge<AsyncInputStream>, size_t bytes) override;

private:
    StringView m_data;
    StreamCloseExpectation m_expectation;
    Vector<size_t> m_chunks;

    bool m_is_closed { false };
    bool m_is_reset { false };
    bool m_is_cancelled { false };

    bool m_encountered_eof { false };
    size_t m_read_head { 0 };
    size_t m_peek_head { 0 };
    size_t m_next_chunk_index { 1 };

    size_t m_last_enqueue { 0 };

    std::coroutine_handle<> m_awaiter;
};

class AsyncMemoryOutputStream final : public AsyncOutputStream {
public:
    // FIXME: Support artificial atomic write limits similar to `chunks` parameter in
    //        AsyncMemoryInputStream.
    AsyncMemoryOutputStream(StreamCloseExpectation expectation);
    ~AsyncMemoryOutputStream();

    virtual void cancel() override;
    virtual Coroutine<void> reset() override;
    virtual Coroutine<ErrorOr<void>> close() override;
    virtual bool is_open() const override;

    virtual Coroutine<ErrorOr<size_t>> write_some(ReadonlyBytes data) override;

    ReadonlyBytes view() const { return m_buffer; }

private:
    StreamCloseExpectation m_expectation;

    bool m_is_closed { false };
    bool m_is_reset { false };
    bool m_is_cancelled { false };

    ByteBuffer m_buffer;
};

Coroutine<ErrorOr<ReadonlyBytes>> read_until_eof(AsyncInputStream& stream);
Vector<size_t> randomly_partition_input(u32 partition_probability_numerator, u32 partition_probability_denominator, size_t length);

}
