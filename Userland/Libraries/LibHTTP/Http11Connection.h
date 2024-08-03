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
class Http11Response;

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

class Http11Response final : public AsyncResource {
public:
    static Coroutine<ErrorOr<NonnullOwnPtr<Http11Response>>> create(Badge<Http11Connection>, RequestData&& data, AsyncConnection<>& connection);

    void reset() override { return m_body->reset(); }
    Coroutine<ErrorOr<void>> close() override { return m_body->close(); }
    bool is_open() const override { return m_body->is_open(); }

    u16 status_code() const { return m_status_code; }
    Vector<Header> const& headers() const { return m_headers; }

    AsyncInputStream& body() { return *m_body; }

private:
    Http11Response(NonnullOwnPtr<AsyncInputStream>&& body, u16 status_code, Vector<Header>&& headers)
        : m_body(move(body))
        , m_status_code(status_code)
        , m_headers(move(headers))
    {
    }

    NonnullOwnPtr<AsyncInputStream> m_body;
    u16 m_status_code { 0 };
    Vector<Header> m_headers;
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
        VERIFY(!m_request_in_flight);
        if (is_open())
            reset();
    }

    void reset() override
    {
        VERIFY(!m_in_critical_section);
        // Unless Http11Connection is in a critical section, the connection is never in an
        // (observable) half-open state.
        m_connection.input->reset();
        m_connection.output->reset();
    }

    Coroutine<ErrorOr<void>> close() override
    {
        VERIFY(!m_request_in_flight);
        auto maybe_error = co_await m_connection.input->close();
        if (maybe_error.is_error()) {
            m_connection.output->reset();
            co_return maybe_error;
        }
        co_return co_await m_connection.output->close();
    }

    bool is_open() const override
    {
        VERIFY(!m_in_critical_section);

        bool result = m_connection.input->is_open();
        VERIFY(result == m_connection.output->is_open());
        return result;
    }

    template<
        typename Func,
        typename T = InvokeResult<Func, Http11Response&>::ReturnType::ResultType>
    Coroutine<ErrorOr<T>> request(RequestData&& data, Func&& func)
    {
        VERIFY(!m_request_in_flight);
        TemporaryChange request_in_flight { m_request_in_flight, true };

        auto response = CO_TRY(co_await Http11Response::create({}, move(data), m_connection));

        // After this point, Http11Response instance is only responsible for resetting the reading
        // end of the connection and, consequently, Http11Connection enters a critical section (as
        // it can't maintain half-openness invariant anymore).
        TemporaryChange in_critical_section { m_in_critical_section, true };

        auto result = co_await func(*response);

        VERIFY(response->is_open() == !result.is_error());
        if (result.is_error()) {
            m_connection.output->reset();
        } else {
            auto maybe_error = co_await response->close();
            if (maybe_error.is_error()) {
                m_connection.output->reset();
                result = maybe_error.release_error();
            }
        }
        co_return result;
    }

private:
    AsyncConnection<> m_connection;
    bool m_request_in_flight { false };
    bool m_in_critical_section { false };
};

}
