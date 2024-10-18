/*
 * Copyright (c) 2024, Dan Klishch <danilklishch@gmail.com>
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <AK/Coroutine.h>
#include <AK/Error.h>
#include <AK/MaybeOwned.h>
#include <AK/Noncopyable.h>
#include <AK/TemporaryChange.h>

namespace AK {

// HybridResource represents a generic resource (e. g. POSIX file descriptor, AsyncStream, HTTP
// response body) with a failible and/or asynchronous destructor. Refer to AsynchronousDesign.md
// documentation page for a description tailored for users of the asynchronous resources.
//
// In order to correctly implement methods of HybridResource, you first have to define (not
// necessarily in code) two abstract operations: Close and Reset. They MUST have the following
// semantics:
//
//  * Close AO:
//     1. Destroy (possibly almost synchronously, if paradigm allows) all background coroutines
//        associated with the resource.
//     2. Shutdown (possibly asynchronously, if paradigm allows) the associated lower-level
//        resource. Shutdown must ensure that if the state of a resource is clean, it will remain so
//        indefinitely. The state cleanness is resource-specific--for example, streams might define
//        it as "no outstanding writes and no unread data".
//     3. Check if the state of the resource is clean. If it is not, call Reset AO and return an
//        error (preferably, EBUSY).
//     4. Free (possibly asynchronously, if paradigm allows) the associated low-level resource.
//     5. Return success.
//
//  * Reset AO:
//     1. Destroy (possibly almost synchronously, if paradigm allows) all background coroutines
//        associated with the resource.
//     2. Free synchronously the associated low-level resource. Preferably, this should be done in
//        a way that cleanly indicates an error for the event producer.
//
// NOTE: If paradigm == Paradigm::Async, Reset AO MUST be almost synchronous, i. e. it MUST resume
//       the caller in the same event loop iteration. In practice, this means that there should be
//       an almost synchronous way to stop and destroy background coroutines of the resource.
//
// In general, you should strive to not allow concurrent awaiters. Implementation practice shows
// that it is almost impossible to correctly synchronize resource state between them. This is an
// explicit requirement for methods defined in HybridResource but you should employ the same "no
// concurrent awaiters" invariant in your own methods.
template<Paradigm paradigm>
class HybridResource {
    AK_MAKE_NONCOPYABLE(HybridResource);
    AK_MAKE_NONMOVABLE(HybridResource);

public:
    HybridResource() = default;

    // Destructor of an HybridResource must perform the following steps when called:
    // 1. If paradigm == Paradigm::Async, assert that nobody is awaiting on the resource.
    // 2. Assert that the resource is not open and not in a critical section.
    virtual ~HybridResource() = default;

    // cancel() must perform the following steps when called:
    // 1. Assert that the resource is not in a critical section (see AsynchronousDesign.md for the
    //    definition of a critical section).
    // 2. If resource is not open, return.
    // 3. Schedule returning ECANCELED from all current resource awaiters. The error MUST be
    //    treated as unrecoverable.
    // 4. If there are no current awaiters, the next attempt to await on the resource (except for
    //    reset()) must result in unrecoverable ECANCELED.
    virtual void cancel() = 0;

    // reset() must perform the following steps when called:
    // 1. Assert that the resource is open, not in a critical section (see AsynchronousDesign.md
    //    for the definition of a critical section), and that nobody is currently awaiting on the
    //    resource.
    // 2. Perform Reset AO.
    virtual WrapIntoCoroutine<paradigm, void> reset() = 0;

    // close() must perform the following steps when called:
    // 1. Assert that the object is fully constructed. For example, a socket might assert that it is
    //    connected.
    // 2. Assert that the resource is open, not in a critical section, and that nobody is currently
    //    awaiting on the resource.
    // 3. Perform Close AO, await (if necessary) and return its result.
    virtual WrapIntoCoroutine<paradigm, ErrorOr<void>> close() = 0;

    // Resource is said to be in an error state if either Reset AO was invoked or if an operation on
    // a resource has failed and an implementation deemed the error unrecoverable. If a resource is
    // being transitioned to an error state because of an internal error, Reset AO (or its
    // equivalent) must be executed by an implementation. Resource is said to be open if it is
    // not in a error state and Close AO has never been called on it.
    //
    // If is_open is called in parallel with another operation, is_open() MAY assert and if doesn't,
    // the result is undefined.
    virtual bool is_open() const = 0;

protected:
    auto guard_async_method()
    {
        VERIFY(is_open());
        return TemporaryChange { m_has_awaiters, true };
    }

    bool m_has_awaiters = false;
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
