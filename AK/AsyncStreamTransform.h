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

template<DerivedFrom<AsyncInputStream> T>
class AsyncStreamTransform : public AsyncInputStream {
public:
    AsyncStreamTransform(MaybeOwned<T>&& stream, AK::Generator<Empty, ErrorOr<void>>&& generator)
        : m_stream(move(stream))
        , m_generator(move(generator))
    {
    }

    ~AsyncStreamTransform()
    {
        VERIFY(m_state != State::Awaiting);
        cancel();
    }

    virtual void cancel() override
    {
        if (m_state == State::Reset)
            return;
        m_state = State::Reset;

        m_stream->cancel();
        if (!m_generator.is_done())
            m_generator.destroy();
    }

    Coroutine<ErrorOr<void>> close() override
    {
        auto _ = guard_method(InternalCall::No);

        if (m_generator.is_done()) {
            m_state = State::Reset;
            if (m_stream.is_owned())
                CO_TRY(co_await m_stream->close());
            co_return {};
        }

        Variant<Empty, ErrorOr<void>> chunk_or_eof = co_await m_generator.next();

        if (chunk_or_eof.has<Empty>()) {
            cancel();
            co_return Error::from_errno(EBUSY);
        }

        m_state = State::Reset;

        auto& error_or_eof = chunk_or_eof.get<ErrorOr<void>>();
        if (error_or_eof.is_error()) {
            VERIFY(!m_stream->is_open());
            co_return error_or_eof.release_error();
        }

        if (m_stream.is_owned())
            CO_TRY(co_await m_stream->close());
        co_return {};
    }

    Coroutine<ErrorOr<bool>> enqueue_some(Badge<AsyncInputStream>) override
    {
        if (m_generator.is_done())
            co_return false;

        Variant<Empty, ErrorOr<void>> chunk_or_eof = co_await m_generator.next();
        if (chunk_or_eof.has<Empty>()) {
            co_return true;
        } else {
            auto& error_or_eof = chunk_or_eof.get<ErrorOr<void>>();
            if (error_or_eof.is_error()) {
                m_state = State::Reset;
                VERIFY(!m_stream->is_open());
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
};

}

#ifdef USING_AK_GLOBALLY
using AK::AsyncStreamTransform;
#endif
