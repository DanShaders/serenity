/*
 * Copyright (c) 2024, Dan Klishch <danilklishch@gmail.com>
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <AK/Coroutine.h>
#include <AK/Variant.h>

namespace AK {

namespace Detail {

class YieldAwaiter {
public:
    YieldAwaiter(std::coroutine_handle<> control_transfer, std::coroutine_handle<>& awaiter)
        : m_control_transfer(control_transfer)
        , m_awaiter(awaiter)
    {
    }

    bool await_ready() const { return false; }

    auto await_suspend(std::coroutine_handle<> handle)
    {
        m_awaiter = handle;
        return m_control_transfer;
    }

    void await_resume() { }

private:
    std::coroutine_handle<> m_control_transfer;
    std::coroutine_handle<>& m_awaiter;
};

template<>
inline constexpr bool IsSynchronouslyAwaitable<YieldAwaiter> = true;

template<typename YieldType, typename ReturnType>
class GeneratorStorage {
    AK_MAKE_NONCOPYABLE(GeneratorStorage);

public:
    enum class CurrentlyStoredType {
        Empty,
        Yield,
        Return,
    };

    GeneratorStorage() = default;

    GeneratorStorage(GeneratorStorage&& other)
    {
        m_currently_stored_type = other.m_currently_stored_type;
        if (m_currently_stored_type == CurrentlyStoredType::Yield) {
            new (m_data) YieldType(move(*reinterpret_cast<YieldType*>(other.m_data)));
        } else if (m_currently_stored_type == CurrentlyStoredType::Return) {
            new (m_data) ReturnType(move(*reinterpret_cast<ReturnType*>(other.m_data)));
        }
        other.destroy_stored_object();
    }

    GeneratorStorage& operator=(GeneratorStorage&& other)
    {
        if (this != &other) {
            this->~GeneratorStorage();
            new (this) GeneratorStorage(move(other));
        }
        return *this;
    }

    void destroy_stored_object()
    {
        switch (m_currently_stored_type) {
        case CurrentlyStoredType::Empty:
            break;
        case CurrentlyStoredType::Yield:
            reinterpret_cast<YieldType*>(m_data)->~YieldType();
            break;
        case CurrentlyStoredType::Return:
            reinterpret_cast<ReturnType*>(m_data)->~ReturnType();
            break;
        }
        m_currently_stored_type = CurrentlyStoredType::Empty;
    }

    template<typename... Args>
    YieldType* place_yield_object(Args&&... args)
    {
        destroy_stored_object();
        m_currently_stored_type = CurrentlyStoredType::Yield;
        return new (m_data) YieldType(forward<Args>(args)...);
    }

    template<typename... Args>
    ReturnType* place_returned_object(Args&&... args)
    {
        destroy_stored_object();
        m_currently_stored_type = CurrentlyStoredType::Return;
        return new (m_data) ReturnType(forward<Args>(args)...);
    }

    CurrentlyStoredType currently_stored_type() const { return m_currently_stored_type; }

    ReturnType& stored_return_value()
    {
        VERIFY(m_currently_stored_type == CurrentlyStoredType::Return);
        return *reinterpret_cast<ReturnType*>(m_data);
    }

    YieldType& stored_yield_value()
    {
        VERIFY(m_currently_stored_type == CurrentlyStoredType::Yield);
        return *reinterpret_cast<YieldType*>(m_data);
    }

    ReturnType* return_value_for_overwriting()
    {
        this->destroy_stored_object();
        m_currently_stored_type = CurrentlyStoredType::Return;
        return reinterpret_cast<ReturnType*>(m_data);
    }

private:
    CurrentlyStoredType m_currently_stored_type = CurrentlyStoredType::Empty;
    alignas(max(alignof(YieldType), alignof(ReturnType))) u8 m_data[max(sizeof(YieldType), sizeof(ReturnType))];
};

}

template<Paradigm paradigm, typename Y, typename R>
class [[nodiscard]] HybridGenerator : private Detail::GeneratorStorage<Y, R> {
    struct GeneratorPromiseType;

    AK_MAKE_NONCOPYABLE(HybridGenerator);

public:
    using YieldType = Y;
    using ReturnType = R;
    using promise_type = GeneratorPromiseType;

    ~HybridGenerator()
    {
        this->destroy_stored_object();
        if (m_handle)
            m_handle.destroy();
    }

    HybridGenerator(HybridGenerator&& other)
    {
        m_handle = AK::exchange(other.m_handle, {});
        m_read_returned_object = exchange(other.m_read_returned_object, false);

        if (m_handle)
            m_handle.promise().m_coroutine = this;
    }

    HybridGenerator& operator=(HybridGenerator&& other)
    {
        if (this != &other) {
            this->~HybridGenerator();
            new (this) HybridGenerator(move(other));
        }
        return *this;
    }

    bool is_done() const { return !m_handle || m_handle.done(); }

    void destroy()
    {
        VERIFY(m_handle && !m_handle.promise().m_awaiter);
        this->destroy_stored_object();
        m_handle.destroy();
        m_handle = {};
    }

    WrapIntoCoroutine<paradigm, Variant<Y, R>> next()
    {
        return [](auto& self) -> CoroutineFacade<paradigm, Variant<Y, R>> {
            if (!self.is_done()) {
                co_await Detail::YieldAwaiter { self.m_handle, self.m_handle.promise().m_awaiter };
                if (self.m_handle)
                    self.m_handle.promise().m_awaiter = {};
            }

            if (self.is_done()) {
                VERIFY(!self.m_read_returned_object);
                self.m_read_returned_object = true;
                co_return move(self.stored_return_value());
            } else {
                co_return move(self.stored_yield_value());
            }
        }(*this);
    }

private:
    template<typename U>
    friend struct Detail::TryAwaiter;

    struct GeneratorPromiseType {
        HybridGenerator get_return_object()
        {
            return { std::coroutine_handle<promise_type>::from_promise(*this) };
        }

        Detail::SuspendAlways initial_suspend() { return {}; }

        Detail::SymmetricControlTransfer final_suspend() noexcept
        {
            VERIFY(m_awaiter);
            return { m_awaiter };
        }

        template<typename U>
        requires IsConstructible<R, U>
        void return_value(U&& returned_object)
        {
            m_coroutine->place_returned_object(forward<U>(returned_object));
        }

        void return_value(ReturnType&& returned_object)
        {
            m_coroutine->place_returned_object(move(returned_object));
        }

        Detail::SymmetricControlTransfer yield_value(YieldType&& yield_value)
        {
            m_coroutine->place_yield_object(move(yield_value));
            VERIFY(m_awaiter);
            return { m_awaiter };
        }

        template<typename U>
        decltype(auto) await_transform(U&& awaitable)
        {
            if constexpr (paradigm == Paradigm::Async || Detail::IsSynchronouslyAwaitable<RemoveCVReference<U>>)
                return forward<U>(awaitable);
            else
                return Detail::WrapIntoAwaitable<U> { awaitable };
        }

        std::coroutine_handle<> m_awaiter;
        HybridGenerator* m_coroutine { nullptr }; // Must be named `m_coroutine` for CO_TRY to work.
    };

    HybridGenerator(std::coroutine_handle<promise_type>&& handle)
        : m_handle(move(handle))
    {
        m_handle.promise().m_coroutine = this;
    }

    ReturnType* return_value() // Must be defined for CO_TRY to work.
    {
        return this->return_value_for_overwriting();
    }

    std::coroutine_handle<promise_type> m_handle;

    bool m_read_returned_object { false };
};

template<typename Y, typename R>
using Generator = HybridGenerator<Paradigm::Async, Y, R>;

template<typename Y, typename R>
using SyncGenerator = HybridGenerator<Paradigm::Sync, Y, R>;

}

#ifdef USING_AK_GLOBALLY
using AK::Generator;
using AK::HybridGenerator;
using AK::SyncGenerator;
#endif
