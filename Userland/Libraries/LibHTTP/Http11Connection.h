/*
 * Copyright (c) 2024, Dan Klishch <danilklishch@gmail.com>
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <AK/AsyncStream.h>
#include <AK/ByteString.h>
#include <AK/NonnullOwnPtr.h>
#include <AK/OwnPtr.h>
#include <AK/TemporaryChange.h>
#include <AK/Vector.h>

namespace HTTP {

class Http11Connection;
struct Http11Response;

#define ENUMERATE_METHODS(F) \
    F(Invalid)               \
    F(HEAD)                  \
    F(GET)                   \
    F(POST)                  \
    F(DELETE)                \
    F(PATCH)                 \
    F(OPTIONS)               \
    F(TRACE)                 \
    F(CONNECT)               \
    F(PUT)

enum class Method {
#define ID(x) x,
    ENUMERATE_METHODS(ID)
#undef ID
};

struct Header {
    ByteString header;
    ByteString value;
};

struct RequestData {
    struct PlainBody {
        StringView data;
    };

    Method method;
    StringView url;
    Vector<Header> headers;
    Variant<Empty, PlainBody> body = Empty {};
};

struct Http11Response {
    static Coroutine<ErrorOr<NonnullOwnPtr<Http11Response>>> create(Badge<Http11Connection>, RequestData&& data, AsyncConnection<>& connection);

    NonnullOwnPtr<AsyncInputStream> body;
    u16 status_code { 0 };
    Vector<Header> headers;
};

class Http11Connection final : public AsyncResource {
public:
    template<typename T, typename U>
    Http11Connection(AsyncConnection<T, U>&& connection)
        : m_connection(move(connection))
    {
    }

    ~Http11Connection()
    {
        VERIFY(m_state != State::Awaiting);
    }

    virtual Coroutine<ErrorOr<void>> close() override
    {
        auto _ = guard_method(InternalCall::No);
        m_state = State::Reset;

        auto maybe_error = co_await m_connection.input->close();
        if (maybe_error.is_error()) {
            m_connection.output->cancel();
            co_return maybe_error.release_error();
        }

        co_return co_await m_connection.output->close();
    }

    template<
        typename Func,
        typename T = InvokeResult<Func, NonnullOwnPtr<Http11Response>>::ReturnType::ResultType>
    Coroutine<ErrorOr<T>> request(RequestData&& data, Func&& func)
    {
        auto _ = guard_method(InternalCall::No);

        ArmedScopeGuard resetter = [&] {
            m_state = State::Reset;
            m_connection.input->cancel();
            m_connection.output->cancel();
        };

        auto response = CO_TRY(co_await Http11Response::create({}, move(data), m_connection));
        auto result = CO_TRY(co_await func(move(response)));

        VERIFY(m_connection.input->is_open());
        resetter.disarm();
        co_return result;
    }

private:
    AsyncConnection<> m_connection;
};

}
