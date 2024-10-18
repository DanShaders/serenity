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

template<DerivedFrom<AsyncResource> T>
class AsyncStreamTransform : public AsyncInputStream {
public:
    AsyncStreamTransform(MaybeOwned<T>&& stream, AK::Generator<Empty, ErrorOr<void>>&& generator)
        : m_stream(move(stream))
        , m_generator(move(generator))
    {
    }

    ~AsyncStreamTransform()
    {
        VERIFY(!is_open() && !m_has_awaiters);
    }

    virtual void cancel() override
    {
        m_stream->cancel();
    }

    virtual Coroutine<void> reset() override
    {
        auto _ = guard_async_method();

        co_await m_stream->reset();
        m_generator.destroy();
        m_is_open = false;
    }

    virtual Coroutine<ErrorOr<void>> close() override
    {
        auto _ = guard_async_method();
        m_is_open = false;

        if (!m_generator.is_done()) {
            Variant<Empty, ErrorOr<void>> chunk_or_eof = co_await m_generator.next();
            if (chunk_or_eof.has<Empty>()) {
                m_generator.destroy();
                co_await m_stream->reset();
                co_return Error::from_errno(EBUSY);
            } else {
                auto& error_or_eof = chunk_or_eof.get<ErrorOr<void>>();
                if (error_or_eof.is_error())
                    co_return error_or_eof.release_error();
            }
        }

        if (m_stream.is_owned())
            CO_TRY(co_await m_stream->close());
        co_return {};
    }

    virtual bool is_open() const override
    {
        return m_is_open;
    }

    virtual Coroutine<ErrorOr<bool>> enqueue_some(Badge<AsyncInputStream>) override
    {
        auto _ = guard_async_method();

        if (m_generator.is_done())
            co_return false;

        Variant<Empty, ErrorOr<void>> chunk_or_eof = co_await m_generator.next();
        if (chunk_or_eof.has<Empty>()) {
            co_return true;
        } else {
            auto& error_or_eof = chunk_or_eof.get<ErrorOr<void>>();
            if (error_or_eof.is_error()) {
                m_is_open = false;
                co_return error_or_eof.release_error();
            } else {
                co_return false;
            }
        }
    }

protected:
    using Generator = AK::Generator<Empty, ErrorOr<void>>;

    MaybeOwned<T> m_stream;

private:
    Generator m_generator;
    bool m_is_open { true };
};

}

#ifdef USING_AK_GLOBALLY
using AK::AsyncStreamTransform;
#endif
