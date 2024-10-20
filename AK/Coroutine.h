/*
 * Copyright (c) 2024, Dan Klishch <danilklishch@gmail.com>
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <AK/Concepts.h>
#include <AK/Noncopyable.h>
#include <coroutine>

namespace AK {

namespace Detail {

// FIXME: GCC ICEs when an implementation of CO_TRY_OR_FAIL with statement expressions is used. See also LibTest/AsyncTestCase.h.
#if defined(AK_COROUTINE_STATEMENT_EXPRS_BROKEN) && !defined(AK_COROUTINE_DESTRUCTION_BROKEN)
#    define AK_USE_TRY_OR_FAIL_AWAITER
#endif

#ifdef AK_USE_TRY_OR_FAIL_AWAITER
namespace Test {

template<typename T>
struct TryOrFailAwaiter;

}
#endif

struct SuspendNever {
    // Even though we set -fno-exceptions, Clang really wants these to be noexcept.
    bool await_ready() const noexcept { return true; }
    void await_suspend(std::coroutine_handle<>) const noexcept { }
    void await_resume() const noexcept { }
};

struct SuspendAlways {
    bool await_ready() const noexcept { return false; }
    void await_suspend(std::coroutine_handle<>) const noexcept { }
    void await_resume() const noexcept { }
};

struct SymmetricControlTransfer {
    SymmetricControlTransfer(std::coroutine_handle<> handle)
        : m_handle(handle ? handle : std::noop_coroutine())
    {
    }

    bool await_ready() const noexcept { return false; }
    auto await_suspend(std::coroutine_handle<>) const noexcept { return m_handle; }
    void await_resume() const noexcept { }

    std::coroutine_handle<> m_handle;
};

template<typename T>
struct TryAwaiter;

template<typename T>
struct ValueHolder {
    alignas(T) u8 m_return_value[sizeof(T)];
};

template<>
struct ValueHolder<void> { };

template<typename T>
inline constexpr bool IsSynchronouslyAwaitable = false;

template<typename T>
inline constexpr bool IsSynchronouslyAwaitable<TryAwaiter<T>> = true;

template<typename U>
struct WrapIntoAwaitable {
    using PointerType = RemoveReference<U>*;

public:
    WrapIntoAwaitable(PointerType expression)
        : m_expression(expression)
    {
    }

    bool await_ready() const { return true; }
    void await_suspend(std::coroutine_handle<>) { VERIFY_NOT_REACHED(); }
    [[nodiscard]] U&& await_resume() { return static_cast<U&&>(*m_expression); }

private:
    PointerType m_expression;
};

}

class SuspendFrame {
public:
    // Suspends the current frame and writes the handle to it to `handle_to_suspend`.
    SuspendFrame(std::coroutine_handle<>& handle_to_suspend)
        : m_handle_to_suspend(handle_to_suspend)
    {
    }

    bool await_ready() { return false; }

    std::coroutine_handle<> await_suspend(std::coroutine_handle<> handle)
    {
        m_handle_to_suspend = handle;
        return {};
    }

    void await_resume() { m_handle_to_suspend = {}; }

private:
    std::coroutine_handle<>& m_handle_to_suspend;
};

class SwapFrames {
public:
    // Resumes `handle_to_resume`, writes the current (suspended) frame into `handle_to_suspend`.
    SwapFrames(std::coroutine_handle<> handle_to_resume, std::coroutine_handle<>& handle_to_suspend)
        : m_handle_to_resume(handle_to_resume)
        , m_handle_to_suspend(handle_to_suspend)
    {
    }

    bool await_ready() { return false; }

    std::coroutine_handle<> await_suspend(std::coroutine_handle<> handle)
    {
        m_handle_to_suspend = handle;
        return m_handle_to_resume ? m_handle_to_resume : std::noop_coroutine();
    }

    void await_resume() { m_handle_to_suspend = {}; }

private:
    std::coroutine_handle<> m_handle_to_resume;
    std::coroutine_handle<>& m_handle_to_suspend;
};

template<typename T>
class [[nodiscard]] Coroutine : private Detail::ValueHolder<T> {
    struct CoroutinePromiseVoid;
    struct CoroutinePromiseValue;

    AK_MAKE_NONCOPYABLE(Coroutine);

public:
    using ReturnType = T;
    using promise_type = Conditional<SameAs<T, void>, CoroutinePromiseVoid, CoroutinePromiseValue>;

    ~Coroutine()
    {
        VERIFY(await_ready());
        if constexpr (!SameAs<T, void>)
            return_value()->~T();
        if (m_handle)
            m_handle.destroy();
    }

    Coroutine(Coroutine&& other)
    {
        m_handle = AK::exchange(other.m_handle, {});
        if (!await_ready())
            m_handle.promise().m_coroutine = this;
        else if constexpr (!IsVoid<T>)
            new (return_value()) T(move(*other.return_value()));
    }

    Coroutine& operator=(Coroutine&& other)
    {
        if (this != &other) {
            this->~Coroutine();
            new (this) Coroutine(move(other));
        }
        return *this;
    }

    bool await_ready() const
    {
        return !m_handle || m_handle.done();
    }

    void await_suspend(std::coroutine_handle<> awaiter)
    {
        m_handle.promise().m_awaiter = awaiter;
    }

    // Do NOT bind the result of await_resume() on a temporary coroutine (or the result of CO_TRY) to auto&&!
    [[nodiscard]] decltype(auto) await_resume()
    {
        if constexpr (SameAs<T, void>)
            return;
        else
            return static_cast<T&&>(*return_value());
    }

private:
    template<typename U>
    friend struct Detail::TryAwaiter;

#ifdef AK_USE_TRY_OR_FAIL_AWAITER
    template<typename U>
    friend struct AK::Detail::Test::TryOrFailAwaiter;
#endif

    // You cannot just have return_value and return_void defined in the same promise type because C++.
    struct CoroutinePromiseBase {
        CoroutinePromiseBase() = default;

        Coroutine get_return_object()
        {
            return { std::coroutine_handle<promise_type>::from_promise(*static_cast<promise_type*>(this)) };
        }

        Detail::SuspendNever initial_suspend() { return {}; }

        Detail::SymmetricControlTransfer final_suspend() noexcept
        {
            return { m_awaiter };
        }

        void unhandled_exception() = delete;

        std::coroutine_handle<> m_awaiter;
        Coroutine* m_coroutine { nullptr };
    };

    struct CoroutinePromiseValue : CoroutinePromiseBase {
        template<typename U>
        requires requires { { T(forward<U>(declval<U>())) }; }
        void return_value(U&& returned_object)
        {
            new (this->m_coroutine->return_value()) T(forward<U>(returned_object));
        }

        void return_value(T&& returned_object)
        {
            new (this->m_coroutine->return_value()) T(move(returned_object));
        }
    };

    struct CoroutinePromiseVoid : CoroutinePromiseBase {
        void return_void() { }
    };

    Coroutine(std::coroutine_handle<promise_type>&& handle)
        : m_handle(move(handle))
    {
        m_handle.promise().m_coroutine = this;
    }

    T* return_value()
    {
        return reinterpret_cast<T*>(this->m_return_value);
    }

    std::coroutine_handle<promise_type> m_handle;
};

template<typename T>
class FakeCoroutine : private Detail::ValueHolder<T> {
    struct CoroutinePromiseVoid;
    struct CoroutinePromiseValue;

    AK_MAKE_NONCOPYABLE(FakeCoroutine);

public:
    using ReturnType = T;
    using promise_type = Conditional<SameAs<T, void>, CoroutinePromiseVoid, CoroutinePromiseValue>;

    ~FakeCoroutine()
    {
        VERIFY(await_ready());
        if constexpr (!IsVoid<T>)
            return_value()->~T();
        if (m_handle)
            m_handle.destroy();
    }

    FakeCoroutine(FakeCoroutine&& other)
    {
        m_handle = AK::exchange(other.m_handle, {});
        VERIFY(await_ready());
        if constexpr (!IsVoid<T>)
            new (return_value()) T(move(*other.return_value()));
    }

    FakeCoroutine& operator=(FakeCoroutine&& other)
    {
        if (this != &other) {
            this->~FakeCoroutine();
            new (this) FakeCoroutine(move(other));
        }
        return *this;
    }

    bool await_ready() const
    {
        VERIFY(!m_handle || m_handle.done());
        return true;
    }

    void await_suspend(std::coroutine_handle<>)
    {
        VERIFY_NOT_REACHED();
    }

    [[nodiscard]] decltype(auto) await_resume()
    {
        if constexpr (SameAs<T, void>)
            return;
        else
            return static_cast<T&&>(*return_value());
    }

    operator decltype(auto)()
    {
        VERIFY(await_ready());
        return await_resume();
    }

private:
    template<typename U>
    friend struct Detail::TryAwaiter;

    // You cannot just have return_value and return_void defined in the same promise type because C++.
    struct CoroutinePromiseBase {
        CoroutinePromiseBase() = default;

        FakeCoroutine get_return_object()
        {
            return { std::coroutine_handle<promise_type>::from_promise(*static_cast<promise_type*>(this)) };
        }

        Detail::SuspendNever initial_suspend() { return {}; }
        Detail::SuspendAlways final_suspend() noexcept { return {}; }

        void unhandled_exception() = delete;

        template<typename U>
        decltype(auto) await_transform(U&& expression)
        {
            if constexpr (Detail::IsSynchronouslyAwaitable<RemoveCVReference<U>>)
                return forward<U>(expression);
            else
                return Detail::WrapIntoAwaitable<U> { &expression };
        }

        FakeCoroutine* m_coroutine { nullptr };
    };

    struct CoroutinePromiseValue : CoroutinePromiseBase {
        template<typename U>
        requires IsConstructible<T, U>
        void return_value(U&& returned_object)
        {
            new (this->m_coroutine->return_value()) T(forward<U>(returned_object));
        }

        void return_value(T&& returned_object)
        {
            new (this->m_coroutine->return_value()) T(move(returned_object));
        }
    };

    struct CoroutinePromiseVoid : CoroutinePromiseBase {
        void return_void() { }
    };

    FakeCoroutine(std::coroutine_handle<promise_type>&& handle)
        : m_handle(move(handle))
    {
        m_handle.promise().m_coroutine = this;
    }

    T* return_value()
    {
        return reinterpret_cast<T*>(this->m_return_value);
    }

    std::coroutine_handle<promise_type> m_handle;
};

template<typename T>
inline constexpr bool Detail::IsSynchronouslyAwaitable<FakeCoroutine<T>> = true;

enum class Paradigm {
    Async,
    Sync,
};

template<Paradigm paradigm, typename T>
using WrapIntoCoroutine = Conditional<paradigm == Paradigm::Async, Coroutine<T>, T>;

template<Paradigm paradigm, typename T>
using CoroutineFacade = Conditional<paradigm == Paradigm::Async, Coroutine<T>, FakeCoroutine<T>>;

template<typename T>
T must_sync(Coroutine<ErrorOr<T>>&& coroutine)
{
    VERIFY(coroutine.await_ready());
    auto&& object = coroutine.await_resume();
    VERIFY(!object.is_error());
    if constexpr (!IsSame<T, void>)
        return object.release_value();
}

#ifndef AK_COROUTINE_DESTRUCTION_BROKEN
namespace Detail {
template<typename T>
struct TryAwaiter {
    TryAwaiter(T& expression)
    requires(!IsSpecializationOf<T, Coroutine>)
        : m_expression(&expression)
    {
    }

    TryAwaiter(T&& expression)
    requires(!IsSpecializationOf<T, Coroutine>)
        : m_expression(&expression)
    {
    }

    bool await_ready() { return false; }

    template<typename U>
    requires IsSpecializationOf<T, ErrorOr>
    std::coroutine_handle<> await_suspend(std::coroutine_handle<U> handle)
    {
        if (!m_expression->is_error()) {
            return handle;
        } else {
            auto* coroutine = handle.promise().m_coroutine;
            using ReturnType = RemoveReference<decltype(*coroutine)>::ReturnType;
            static_assert(IsSpecializationOf<ReturnType, ErrorOr>,
                "CO_TRY can only be used inside functions returning a specialization of ErrorOr");

            // Move error to the user-visible AK::Coroutine
            new (coroutine->return_value()) ReturnType(m_expression->release_error());
            // ... and tell it that there's a result available.
            coroutine->m_handle = {};

            // Run destructors for locals in the coroutine that failed.
            handle.destroy();

            // Lastly, transfer control to the parent (or nothing, if parent is not yet suspended).
            if constexpr (requires { handle.promise().m_awaiter; }) {
                auto awaiter = handle.promise().m_awaiter;
                if (awaiter)
                    return awaiter;
            }
            return std::noop_coroutine();
        }
    }

    decltype(auto) await_resume()
    {
        return m_expression->release_value();
    }

    T* m_expression { nullptr };
};
}

#    ifndef AK_COROUTINE_TYPE_DEDUCTION_BROKEN
#        define CO_TRY(expression) (co_await ::AK::Detail::TryAwaiter { (expression) })
#    else
namespace Detail {
template<typename T>
auto declval_coro_result(Coroutine<T>&&) -> T;
template<typename T>
auto declval_coro_result(T&&) -> T;
}

// GCC cannot handle CO_TRY(...CO_TRY(...)...), this hack ensures that it always has the right type information available.
// FIXME: Remove this once GCC can correctly infer the result type of `co_await TryAwaiter { ... }`.
#        define CO_TRY(expression) static_cast<decltype(AK::Detail::declval_coro_result(expression).release_value())>(co_await ::AK::Detail::TryAwaiter { (expression) })
#    endif
#elifndef AK_COROUTINE_STATEMENT_EXPRS_BROKEN
#    define CO_TRY(expression)                               \
        ({                                                   \
            AK_IGNORE_DIAGNOSTIC("-Wshadow",                 \
                auto _temporary_result = (expression));      \
            if (_temporary_result.is_error()) [[unlikely]]   \
                co_return _temporary_result.release_error(); \
            _temporary_result.release_value();               \
        })
#else
#    error Unable to work around compiler bugs in definiton of CO_TRY.
#endif

}

#ifdef USING_AK_GLOBALLY
using AK::Coroutine;
#endif
