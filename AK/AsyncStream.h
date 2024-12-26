/*
 * Copyright (c) 2024, Dan Klishch <danilklishch@gmail.com>
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <AK/Badge.h>
#include <AK/Resource.h>

namespace AK {

// AsyncInputStream is a base class for all asynchronous input streams. Refer to
// AsynchronousDesign.md documentation page for a description tailored for users of the streams.
//
// In order to implement a brand new AsyncInputStream, you generally have to define a destructor and
// overload six virtual functions: 3 from AsyncResource and 3 from AsyncInputStream. When
// implementing the AsyncResource interface, please note that AsyncInputStream is considered clean
// if there's no data left to be read.
template<Paradigm paradigm>
class HybridInputStream : public virtual HybridResource<paradigm> {
public:
    struct PeekOrEofResult {
        ReadonlyBytes data;
        bool is_eof;
    };

    HybridInputStream() = default;

    virtual void cancel() = 0;

    ReadonlyBytes buffered_data() const
    {
        VERIFY(this->is_unlocked());
        return buffered_data_unchecked({});
    }

    WrapIntoCoroutine<paradigm, ErrorOr<PeekOrEofResult>> peek_or_eof()
    {
        return peek_or_eof(InternalCall::No);
    }

    WrapIntoCoroutine<paradigm, ErrorOr<ReadonlyBytes>> peek()
    {
        return peek(InternalCall::No);
    }

    WrapIntoCoroutine<paradigm, ErrorOr<ReadonlyBytes>> read(size_t bytes)
    {
        return read(InternalCall::No, bytes);
    }

    template<typename T>
    WrapIntoCoroutine<paradigm, ErrorOr<T>> read_object()
    {
        return read_object(InternalCall::No);
    }

    // If EOF has not been reached, `enqueue_some` should read at least one byte from the underlying
    // stream to the internal buffer and return true. Otherwise, it must not change the buffer and
    // return false. If read fails and, consequently, `enqueue_some` returns Error, it must
    // perform Reset AO (or an equivalent of it). Therefore, all reading errors are considered fatal
    // for AsyncInputStream. Additionally, implementation must assert if `enqueue_some` is called
    // concurrently. This is the only method that can be interrupted by `reset`.
    virtual WrapIntoCoroutine<paradigm, ErrorOr<bool>> enqueue_some(Badge<HybridInputStream>) = 0;

    // `buffered_data_unchecked` should just return a view of the buffer. It must not invalidate
    // previously returned views of the buffer.
    virtual ReadonlyBytes buffered_data_unchecked(Badge<HybridInputStream>) const = 0;

    // `dequeue` should remove `bytes` bytes from the buffer. It is guaranteed that this amount of
    // bytes will be present in the buffer at the point of the call. `dequeue` must not invalidate
    // previously returned views of the buffer. There are some restrictions on `bytes` parameter
    // originating from the length condition (see documentation), so if you just use
    // AsyncStreamBuffer as the stream buffer, `dequeue` and `enqueue_some` will have amortized
    // O(stream_length) complexity.
    virtual void dequeue(Badge<HybridInputStream>, size_t bytes) = 0;

protected:
    using InternalCall = HybridResource<paradigm>::InternalCall;

    static Badge<HybridInputStream> badge() { return {}; }

    bool m_is_reading_peek { false };

private:
    CoroutineFacade<paradigm, ErrorOr<PeekOrEofResult>> peek_or_eof(InternalCall is_internal_call)
    {
        auto _ = this->guard_method(is_internal_call);

        if (!m_is_reading_peek) {
            m_is_reading_peek = true;
            auto data = buffered_data_unchecked({});
            if (!data.is_empty())
                co_return PeekOrEofResult { data, false };
        }

        bool is_not_eof = CO_TRY(co_await enqueue_some({}));
        co_return PeekOrEofResult { buffered_data_unchecked({}), !is_not_eof };
    }

    CoroutineFacade<paradigm, ErrorOr<ReadonlyBytes>> peek(InternalCall is_internal_call)
    {
        auto _ = this->guard_method(is_internal_call);

        auto [data, is_eof] = CO_TRY(co_await peek_or_eof(InternalCall::Yes));
        if (is_eof) {
            this->cancel();
            co_return Error::from_errno(EIO);
        }
        co_return data;
    }

    CoroutineFacade<paradigm, ErrorOr<ReadonlyBytes>> read(InternalCall is_internal_call, size_t bytes)
    {
        auto _ = this->guard_method(is_internal_call);

        m_is_reading_peek = false;

        if (bytes) {
            auto buffer = buffered_data_unchecked({});
            while (buffer.size() < bytes) {
                if (!CO_TRY(co_await enqueue_some({}))) {
                    this->cancel();
                    co_return Error::from_errno(EIO);
                }
                buffer = buffered_data_unchecked({});
            }
            dequeue({}, bytes);
            co_return buffer.slice(0, bytes);
        } else {
            co_return Bytes {};
        }
    }

    template<typename T>
    CoroutineFacade<paradigm, ErrorOr<T>> read_object(InternalCall is_internal_call)
    {
        auto _ = this->guard_method(is_internal_call);

        auto bytes = CO_TRY(co_await read(InternalCall::Yes, sizeof(T)));
        union {
            T object;
            char representation[sizeof(T)];
        } reinterpreter = {};
        memcpy(&reinterpreter, bytes.data(), sizeof(T));
        co_return reinterpreter.object;
    }
};

using AsyncInputStream = HybridInputStream<Paradigm::Async>;
using SyncInputStream = HybridInputStream<Paradigm::Sync>;

template<Paradigm paradigm>
class HybridOutputStream : public virtual HybridResource<paradigm> {
public:
    HybridOutputStream() = default;

    virtual void cancel() = 0;

    virtual WrapIntoCoroutine<paradigm, ErrorOr<size_t>> write_some(ReadonlyBytes buffer) = 0;

    virtual WrapIntoCoroutine<paradigm, ErrorOr<void>> write(ReadonlySpan<ReadonlyBytes> buffers)
    {
        return [](auto& self, ReadonlySpan<ReadonlyBytes> buffers) -> CoroutineFacade<paradigm, ErrorOr<void>> {
            for (auto buffer : buffers) {
                while (!buffer.is_empty()) {
                    auto nwritten = CO_TRY(co_await self.write_some(buffer));
                    buffer = buffer.slice(nwritten);
                }
            }
            co_return {};
        }(*this, buffers);
    }
};

using AsyncOutputStream = HybridOutputStream<Paradigm::Async>;
using SyncOutputStream = HybridOutputStream<Paradigm::Sync>;

template<DerivedFrom<AsyncInputStream> InputStream = AsyncInputStream, DerivedFrom<AsyncOutputStream> OutputStream = AsyncOutputStream>
struct AsyncConnection {
    operator AsyncConnection<>() &&
    {
        return {
            input.template release_nonnull<AsyncInputStream>(),
            output.template release_nonnull<AsyncOutputStream>(),
        };
    }

    NonnullOwnPtr<InputStream> input;
    NonnullOwnPtr<OutputStream> output;
};

}

#ifdef USING_AK_GLOBALLY
using AK::AsyncConnection;
using AK::AsyncInputStream;
using AK::AsyncOutputStream;
using AK::AsyncResource;
#endif
