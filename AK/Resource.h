/*
 * Copyright (c) 2024, Dan Klishch <danilklishch@gmail.com>
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <AK/Coroutine.h>
#include <AK/Error.h>
#include <AK/MaybeOwned.h>
#include <AK/Noncopyable.h>
#include <AK/ScopeGuard.h>

namespace AK {

template<Paradigm paradigm>
class HybridResource {
    AK_MAKE_NONCOPYABLE(HybridResource);
    AK_MAKE_NONMOVABLE(HybridResource);

public:
    HybridResource() = default;

    virtual ~HybridResource() = default;

    virtual WrapIntoCoroutine<paradigm, ErrorOr<void>> close() = 0;

    bool is_open() const { return m_state != State::Reset; }

protected:
    enum class State {
        Ready,
        Awaiting,
        Reset,
    };

    enum class InternalCall {
        Yes,
        No,
    };

    bool is_unlocked() const { return m_state == State::Ready; }

    auto guard_method(InternalCall is_internal_call)
    {
        if (is_internal_call == InternalCall::Yes) {
            VERIFY(m_state == State::Awaiting);
        } else {
            VERIFY(m_state == State::Ready);
            m_state = State::Awaiting;
        }
        return ScopeGuard { [&] {
            if (is_internal_call != InternalCall::Yes && m_state == State::Awaiting)
                m_state = State::Ready;
        } };
    }

    State m_state = State::Ready;
};

using AsyncResource = HybridResource<Paradigm::Async>;
using SyncResource = HybridResource<Paradigm::Sync>;

}

#ifdef USING_AK_GLOBALLY
using AK::AsyncResource;
using AK::HybridResource;
using AK::Paradigm;
using AK::SyncResource;
#endif
