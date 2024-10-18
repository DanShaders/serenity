/*
 * Copyright (c) 2024, Dan Klishch <danilklishch@gmail.com>
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <AK/Random.h>
#include <AK/TemporaryChange.h>
#include <LibTest/AsyncTestStreams.h>

namespace Test {

static auto s_deferred_context = Core::DeferredInvocationContext::construct();

struct Spinner {
    Spinner(std::coroutine_handle<>& awaiter)
        : m_awaiter(awaiter)
    {
        VERIFY(!m_awaiter);
    }

    bool await_ready() const { return false; }

    void await_suspend(std::coroutine_handle<> awaiter)
    {
        m_awaiter = awaiter;
        Core::ThreadEventQueue::current().post_event(
            s_deferred_context,
            make<Core::DeferredInvocationEvent>(s_deferred_context, [&] {
                m_awaiter.resume();
            }));
    }

    void await_resume() { m_awaiter = {}; }

    std::coroutine_handle<>& m_awaiter;
};

AsyncMemoryInputStream::AsyncMemoryInputStream(StringView data, StreamCloseExpectation expectation, Vector<size_t>&& chunks)
    : m_data(data)
    , m_expectation(expectation)
    , m_chunks(move(chunks))
    , m_peek_head(m_chunks[0])
{
    size_t accumulator = 0;
    for (auto& value : m_chunks) {
        accumulator += value;
        value = accumulator;
    }
    VERIFY(accumulator == m_data.length());
}

AsyncMemoryInputStream::~AsyncMemoryInputStream()
{
    VERIFY(!is_open() && !m_has_awaiters);

    if (m_expectation == StreamCloseExpectation::Reset) {
        EXPECT(m_is_reset);
    } else if (m_expectation == StreamCloseExpectation::Close) {
        EXPECT(m_is_closed);
    } else {
        VERIFY_NOT_REACHED();
    }
}

void AsyncMemoryInputStream::cancel()
{
    if (!is_open())
        return;

    m_is_cancelled = true;
}

Coroutine<void> AsyncMemoryInputStream::reset()
{
    auto _ = guard_async_method();

    VERIFY(is_open());
    m_is_reset = true;
    co_return;
}

Coroutine<ErrorOr<void>> AsyncMemoryInputStream::close()
{
    auto _ = guard_async_method();

    if (m_read_head != m_data.length()) {
        m_is_reset = true;
        co_return Error::from_errno(EBUSY);
    }

    m_is_closed = true;
    co_return {};
}

bool AsyncMemoryInputStream::is_open() const
{
    return !m_is_closed && !m_is_reset;
}

Coroutine<ErrorOr<bool>> AsyncMemoryInputStream::enqueue_some(Badge<AsyncInputStream>)
{
    auto _ = guard_async_method();

    if (m_is_cancelled) {
        m_is_reset = true;
        co_return Error::from_errno(ECANCELED);
    }

    if (m_next_chunk_index == m_chunks.size()) {
        m_last_enqueue = m_peek_head;
        co_return false;
    }

    co_await Spinner { m_awaiter };
    if (m_is_cancelled) {
        m_is_reset = true;
        co_return Error::from_errno(ECANCELED);
    }

    m_last_enqueue = m_peek_head;
    m_peek_head = m_chunks[m_next_chunk_index++];
    co_return true;
}

ReadonlyBytes AsyncMemoryInputStream::buffered_data_unchecked(Badge<AsyncInputStream>) const
{
    return m_data.bytes().slice(m_read_head, m_peek_head - m_read_head);
}

void AsyncMemoryInputStream::dequeue(Badge<AsyncInputStream>, size_t bytes)
{
    m_read_head += bytes;
    VERIFY(m_last_enqueue <= m_read_head && m_read_head <= m_peek_head);
}

AsyncMemoryOutputStream::AsyncMemoryOutputStream(StreamCloseExpectation expectation)
    : m_expectation(expectation)
{
}

AsyncMemoryOutputStream::~AsyncMemoryOutputStream()
{
    VERIFY(!is_open());

    if (m_expectation == StreamCloseExpectation::Reset) {
        EXPECT(m_is_reset);
    } else if (m_expectation == StreamCloseExpectation::Close) {
        EXPECT(m_is_closed);
    } else {
        VERIFY_NOT_REACHED();
    }
}

void AsyncMemoryOutputStream::cancel()
{
    if (!is_open())
        return;

    m_is_cancelled = true;
}

Coroutine<void> AsyncMemoryOutputStream::reset()
{
    VERIFY(is_open());
    m_is_reset = true;
    co_return;
}

Coroutine<ErrorOr<void>> AsyncMemoryOutputStream::close()
{
    VERIFY(is_open());
    m_is_closed = true;
    co_return {};
}

bool AsyncMemoryOutputStream::is_open() const
{
    return !m_is_closed && !m_is_reset;
}

Coroutine<ErrorOr<size_t>> AsyncMemoryOutputStream::write_some(ReadonlyBytes data)
{
    VERIFY(is_open());
    m_buffer.append(data);
    co_return data.size();
}

Coroutine<ErrorOr<ReadonlyBytes>> read_until_eof(AsyncInputStream& stream)
{
    size_t previously_returned_size = 0;
    while (true) {
        auto [data, is_eof] = CO_TRY(co_await stream.peek_or_eof());

        EXPECT(is_eof || previously_returned_size < data.size());
        previously_returned_size = data.size();

        if (is_eof) {
            auto result = must_sync(stream.read(data.size()));

            // Poke stream one more time just to be sure :^)
            auto [empty_data, set_eof_flag] = CO_TRY(co_await stream.peek_or_eof());
            EXPECT(empty_data.is_empty());
            EXPECT(set_eof_flag);

            co_return result;
        }
    }
}

Vector<size_t> randomly_partition_input(u32 partition_probability_numerator, u32 partition_probability_denominator, size_t length)
{
    Vector<size_t> result { 0 };
    for (size_t i = 0; i < length; ++i) {
        if (AK::get_random_uniform(partition_probability_denominator) < partition_probability_numerator) {
            result.append(1);
        } else {
            ++result.last();
        }
    }
    return result;
}

}
