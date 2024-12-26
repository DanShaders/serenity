/*
 * Copyright (c) 2024, Dan Klishch <danilklishch@gmail.com>
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <AK/Random.h>
#include <LibTest/AsyncTestStreams.h>

namespace Test {

struct Spinner {
    Spinner(std::coroutine_handle<>& awaiter, NonnullRefPtr<Core::DeferredInvocationContext> const& context)
        : m_awaiter(awaiter)
        , m_context(context)
    {
        VERIFY(!m_awaiter);
    }

    bool await_ready() const { return false; }

    void await_suspend(std::coroutine_handle<> awaiter)
    {
        m_awaiter = awaiter;
        Core::ThreadEventQueue::current().post_event(
            m_context,
            make<Core::DeferredInvocationEvent>(m_context, [&] {
                m_awaiter.resume();
            }));
    }

    void await_resume() { m_awaiter = {}; }

    std::coroutine_handle<>& m_awaiter;
    NonnullRefPtr<Core::DeferredInvocationContext> m_context;
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
    VERIFY(m_state != State::Awaiting);
    cancel();
}

void AsyncMemoryInputStream::cancel()
{
    if (m_state == State::Reset)
        return;
    m_state = State::Reset;

    EXPECT(m_expectation == StreamCloseExpectation::Reset);

    if (m_awaiter)
        m_awaiter.resume();
}

Coroutine<ErrorOr<void>> AsyncMemoryInputStream::close()
{
    auto _ = guard_method(InternalCall::No);

    if (m_read_head != m_data.length()) {
        cancel();
        co_return Error::from_errno(EBUSY);
    }

    EXPECT(m_expectation == StreamCloseExpectation::Close);
    m_state = State::Reset;
    co_return {};
}

Coroutine<ErrorOr<bool>> AsyncMemoryInputStream::enqueue_some(Badge<AsyncInputStream>)
{
    if (m_next_chunk_index == m_chunks.size()) {
        m_last_enqueue = m_peek_head;
        co_return false;
    }

    {
        auto context = Core::DeferredInvocationContext::construct();
        co_await Spinner { m_awaiter, context };
    }

    if (m_state == State::Reset)
        co_return Error::from_errno(ECANCELED);

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
    VERIFY(m_state != State::Awaiting);
    cancel();
}

void AsyncMemoryOutputStream::cancel()
{
    if (m_state == State::Reset)
        return;
    m_state = State::Reset;

    EXPECT(m_expectation == StreamCloseExpectation::Reset);
}

Coroutine<ErrorOr<void>> AsyncMemoryOutputStream::close()
{
    auto _ = guard_method(InternalCall::No);

    EXPECT(m_expectation == StreamCloseExpectation::Close);
    m_state = State::Reset;
    co_return {};
}

Coroutine<ErrorOr<size_t>> AsyncMemoryOutputStream::write_some(ReadonlyBytes data)
{
    auto _ = guard_method(InternalCall::No);
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
