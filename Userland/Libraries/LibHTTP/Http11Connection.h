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
    // Does not consistently reset AsyncConnection on error.
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
        VERIFY(!is_open() && !m_has_awaiters);
    }

    virtual void cancel() override
    {
        VERIFY(!m_in_critical_section);
        m_connection.input->cancel();
        m_connection.output->cancel();
    }

    virtual Coroutine<void> reset() override
    {
        auto _ = guard_async_method();

        // Unless another operation is in progress, the connection is never in an half-open state.
        co_await m_connection.input->reset();
        co_await m_connection.output->reset();
    }

    virtual Coroutine<ErrorOr<void>> close() override
    {
        auto _ = guard_async_method();

        auto maybe_error = co_await m_connection.input->close();
        if (maybe_error.is_error()) {
            co_await m_connection.output->reset();
            co_return maybe_error;
        }
        co_return co_await m_connection.output->close();
    }

    virtual bool is_open() const override
    {
        bool result = m_connection.input->is_open();
        // This is only true if !m_has_awaiters. We leverage UB allowed by the interface here.
        VERIFY(result == m_connection.output->is_open());
        return result;
    }

    template<
        typename Func,
        typename T = InvokeResult<Func, NonnullOwnPtr<Http11Response>&&>::ReturnType::ResultType>
    Coroutine<ErrorOr<T>> request(RequestData&& data, Func&& func)
    {
        auto _ = guard_async_method();

        auto response = co_await Http11Response::create({}, move(data), m_connection);

        if (response.is_error()) {
            if (m_connection.input->is_open())
                co_await m_connection.input->reset();
            if (m_connection.output->is_open())
                co_await m_connection.output->reset();
            co_return response.release_error();
        }

        // After this point, Http11Response instance is only responsible for resetting the reading
        // end of the connection and, consequently, Http11Connection enters a critical section.
        m_in_critical_section = true;
        auto result = co_await func(response.release_value());
        m_in_critical_section = false;

        VERIFY(response.value().leak_ptr() == nullptr);

        if (result.is_error())
            co_await m_connection.output->reset();
        co_return result;
    }

private:
    AsyncConnection<> m_connection;
    bool m_in_critical_section { false };
};

}
