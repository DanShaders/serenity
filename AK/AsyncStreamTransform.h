/*
 * Copyright (c) 2024, Dan Klishch <danilklishch@gmail.com>
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <AK/AsyncStream.h>
#include <AK/Generator.h>
#include <AK/MaybeOwned.h>
#include <AK/TemporaryChange.h>

namespace AK {

template<DerivedFrom<StreamBuffer> Buffer>
class AsyncStreamTransform;

namespace Detail {

template<typename T>
class ResumeOtherContinuation {
public:
    // Resumes `handle_to_resume` and makes the current frame the awaiter of the `continuation`, so
    // that when `continuation` gets resolved, it wakes us up again.
    ResumeOtherContinuation(std::coroutine_handle<> handle_to_resume, Coroutine<T>& continuation)
        : m_handle_to_resume(handle_to_resume)
        , m_continuation(continuation)
    {
    }

    bool await_ready() { return false; }

    std::coroutine_handle<> await_suspend(std::coroutine_handle<> handle)
    {
        m_continuation.await_suspend(handle);
        return m_handle_to_resume ? m_handle_to_resume : std::noop_coroutine();
    }

    decltype(auto) await_resume()
    {
        return m_continuation.await_resume();
    }

private:
    std::coroutine_handle<> m_handle_to_resume;
    Coroutine<T>& m_continuation;
};

class WaitForProgress {
public:
    // Resumes `handle_to_resume` (that should do progress on coroutine_to_wait), writes the
    // current (suspended) frame into `handle_to_suspend` (so that it can be woken up from somewhere
    // else; in practice, from ResumeOtherContinuation when we detect that no progress can be made
    // and we should yield to the event loop), and into `coroutine_to_wait` (so that the resolution
    // of the coroutine wakes up the frame).
    WaitForProgress(std::coroutine_handle<> handle_to_resume, std::coroutine_handle<>& handle_to_suspend, Coroutine<ErrorOr<void>>& coroutine_to_wait)
        : m_handle_to_resume(handle_to_resume)
        , m_handle_to_suspend(handle_to_suspend)
        , m_coroutine_to_wait(coroutine_to_wait)
    {
    }

    bool await_ready() { return false; }

    std::coroutine_handle<> await_suspend(std::coroutine_handle<> handle)
    {
        m_handle_to_suspend = handle;
        m_coroutine_to_wait.await_suspend(handle);
        return m_handle_to_resume ? m_handle_to_resume : std::noop_coroutine();
    }

    void await_resume()
    {
        m_coroutine_to_wait.await_suspend({});
        m_handle_to_suspend = {};
    }

private:
    std::coroutine_handle<> m_handle_to_resume;
    std::coroutine_handle<>& m_handle_to_suspend;
    Coroutine<ErrorOr<void>>& m_coroutine_to_wait;
};

// AsyncStreamTransform needs to wrap the underlying stream into one more level of indirection to
// detect when an operation on the stream will block the generator and resume its internal frame
// when this happens. See comment near AsyncStreamTransform for more information.
class StreamWrapper final : public AsyncInputStream {
public:
    StreamWrapper(MaybeOwned<AsyncInputStream>&& stream, std::coroutine_handle<>& continuation_on_stream_block)
        : m_stream(move(stream))
        , m_continuation_on_stream_block(continuation_on_stream_block)
    {
    }

    virtual void cancel() override
    {
        if (m_is_attached)
            m_stream->cancel();
    }

    virtual Coroutine<void> reset() override
    {
        auto _ = guard_async_method();
        m_is_attached = false;

        // `reset` should be "fast enough" as per interface specification, so no need to check if it
        // blocks `enqueue_some`.
        return m_stream->reset();
    }

    virtual Coroutine<ErrorOr<void>> close() override
    {
        // User-provided generator should treat StreamWrapper embedded to AsyncStreamTransform as
        // non-owned and thus shouldn't call close on it.
        VERIFY_NOT_REACHED();
    }

    virtual bool is_open() const override
    {
        return m_is_attached && m_stream->is_open();
    }

    virtual Coroutine<ErrorOr<bool>> enqueue_some(Badge<AsyncInputStream>) override
    {
        auto _ = guard_async_method();

        auto future = m_stream->enqueue_some(badge());
        if (future.await_ready())
            co_return future.await_resume();

        // If m_continuation_on_stream_block is not null, it resumes wait_for_new_data.
        co_return co_await ResumeOtherContinuation { m_continuation_on_stream_block, future };
    }

    virtual ReadonlyBytes buffered_data_unchecked(Badge<AsyncInputStream>) const override
    {
        return m_stream->buffered_data_unchecked(badge());
    }

    virtual void dequeue(Badge<AsyncInputStream>, size_t bytes) override
    {
        m_stream->dequeue(badge(), bytes);
    }

    template<typename T>
    MaybeOwned<AsyncInputStream>& underlying_stream(Badge<AsyncStreamTransform<T>>) { return m_stream; }

private:
    bool m_is_attached { true };
    MaybeOwned<AsyncInputStream> m_stream;
    std::coroutine_handle<>& m_continuation_on_stream_block;
};

}

// AsyncStreamTransform provides a way to wrap a coroutine that reads from an AsyncInputStream and
// writes some bytes based on read data into another AsyncInputStream. Generally, the generator
// coroutine is let run and create a new stream chuck for as long as it can read data from the
// underlying stream synchronously. AsyncStreamTransform automatically detects when the read
// operation cannot be satisfied synchronously, breaks the current chunk, and passes it to the
// higher-level data consumer. Additionally, the generating coroutine can voluntarily break up the
// current stream chunk and yield to the consumer by calling and awaiting protected `yield` method.
//
// Note that the generating coroutine can make progress even when nobody is waiting on
// AsyncStreamTransform's enqueue_some. It cannot, though, write more data to the buffer as this
// would invalidate stream's data views. This is why buffer access is protected using
// `with_buffer` function.
//
// In order to detect when the stream operation blocks, the underlying stream is wrapped into
// StreamWrapper class that juggles frames when read cannot be satisfied immediately. This juggle
// results in the chunk splitting behavior outlined above.
template<DerivedFrom<StreamBuffer> Buffer>
class AsyncStreamTransform : public AsyncInputStream {
    enum class Status {
        NewData,
        EOFEncountered,
        Error,
    };

public:
    AsyncStreamTransform(MaybeOwned<AsyncInputStream>&& stream)
        : m_stream(move(stream), m_enqueue_some_continuation)
    {
    }

    ~AsyncStreamTransform()
    {
        VERIFY(!is_open() && !m_has_awaiters && m_generator->await_ready());
    }

    virtual void cancel() override final
    {
        if (m_is_open)
            m_stream.cancel();
    }

    virtual Coroutine<void> reset() override final
    {
        auto _ = guard_async_method();
        m_is_open = false;

        m_stream.cancel();
        VERIFY(co_await wait_for_new_data() != Status::NewData);

        if (m_stream.is_open())
            co_await m_stream.reset();
    }

    virtual Coroutine<ErrorOr<void>> close() override final
    {
        auto _ = guard_async_method();

        Status status = co_await wait_for_new_data();

        // Make sure to set m_is_open to false _after_ wait_for_new_data is called because otherwise
        // calls to with_buffer will fail with ECANCELED even though they shouldn't.
        m_is_open = false;

        if (status == Status::NewData) {
            m_stream.cancel();
            VERIFY(co_await wait_for_new_data() != Status::NewData);
            if (m_stream.is_open())
                co_await m_stream.reset();
            // We can potentially discard an error from the generator here but this is fine as the
            // earliest erroneous thing to happen was generator writing more data when none was
            // expected.
            co_return Error::from_errno(EBUSY);
        } else if (status == Status::EOFEncountered) {
            auto& underlying_stream = m_stream.underlying_stream(Badge<AsyncStreamTransform> {});
            if (underlying_stream.is_owned())
                CO_TRY(co_await underlying_stream->close());
            co_return {};
        } else {
            if (m_stream.is_open())
                co_await m_stream.reset();
            co_return m_generator->await_resume().release_error();
        }
    }

    virtual bool is_open() const override final
    {
        return m_is_open;
    }

    virtual Coroutine<ErrorOr<bool>> enqueue_some(Badge<AsyncInputStream>) override final
    {
        auto _ = guard_async_method();

        Status status = co_await wait_for_new_data();

        if (status == Status::NewData) {
            co_return true;
        } else if (status == Status::EOFEncountered) {
            co_return false;
        } else {
            m_is_open = false;
            if (m_stream.is_open())
                co_await m_stream.reset();
            co_return m_generator->await_resume().release_error();
        }
    }

    virtual ReadonlyBytes buffered_data_unchecked(Badge<AsyncInputStream>) const override final
    {
        return m_buffer.data();
    }

    virtual void dequeue(Badge<AsyncInputStream>, size_t bytes) override final
    {
        m_buffer.dequeue(bytes);
    }

protected:
    // Does not need to reset m_stream on error.
    virtual Coroutine<ErrorOr<void>> generate() = 0;

    Coroutine<void> yield()
    {
        co_await SwapFrames { m_enqueue_some_continuation, m_generator_continuation };
    }

    template<typename Func>
    Coroutine<ErrorOr<void>> with_buffer(Func&& func)
    {
        if (!m_is_buffer_unlocked) {
            co_await SuspendFrame { m_generator_continuation };
            // We are woken up by wait_for_new_data.
            VERIFY(m_is_buffer_unlocked);
        }

        if (!m_is_open)
            co_return Error::from_errno(ECANCELED);

        if constexpr (SameAs<InvokeResult<Func, Buffer&>, void>) {
            func(m_buffer);
        } else {
            static_assert(SameAs<InvokeResult<Func, Buffer&>, ErrorOr<void>>);
            CO_TRY(func(m_buffer));
        }

        if (m_buffer.data().size() - m_buffer_size_at_chunk_start > PREFERRED_CHUNK_SIZE)
            co_await yield();

        co_return {};
    }

    // Should be treated as non-owned reference in child classes.
    Detail::StreamWrapper m_stream;

private:
    Coroutine<Status> wait_for_new_data()
    {
        TemporaryChange buffer_unlocker { m_is_buffer_unlocked, true };

        m_buffer_size_at_chunk_start = m_buffer.data().size();

        if (!m_generator.has_value()) {
            m_generator = generate();
        }

        while (!m_generator->await_ready() && m_buffer_size_at_chunk_start == m_buffer.data().size())
            co_await Detail::WaitForProgress { m_generator_continuation, m_enqueue_some_continuation, m_generator.value() };

        if (m_buffer_size_at_chunk_start != m_buffer.data().size()) {
            co_return Status::NewData;
        } else if (!m_generator->await_resume().is_error()) {
            co_return Status::EOFEncountered;
        } else {
            co_return Status::Error;
        }
    }

    std::coroutine_handle<> m_enqueue_some_continuation;
    std::coroutine_handle<> m_generator_continuation;

    Optional<Coroutine<ErrorOr<void>>> m_generator;
    size_t m_buffer_size_at_chunk_start { 0 };
    Buffer m_buffer;
    bool m_is_buffer_unlocked { false };
    bool m_is_open { true };
};

}

#ifdef USING_AK_GLOBALLY
using AK::AsyncStreamTransform;
#endif
