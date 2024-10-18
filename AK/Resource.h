/*
 * Copyright (c) 2024, Dan Klishch <danilklishch@gmail.com>
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <AK/Coroutine.h>
#include <AK/Error.h>
#include <AK/MaybeOwned.h>
#include <AK/Noncopyable.h>

namespace AK {

// HybridResource represents a generic resource (e. g. POSIX file descriptor, AsyncStream, HTTP
// response body) with a failible and/or asynchronous destructor. Refer to AsynchronousDesign.md
// documentation page for a description tailored for users of the asynchronous resources.
//
// In order to correctly implement methods of HybridResource, you first have to define (not
// necessarily in code) two abstract operations: Close and Reset. They should have the following
// semantics:
//
//  * Close AO:
//     1. If paradigm == Paradigm::Async, assert that nobody is awaiting on a resource.
//     2. Ensure that further attempts to wait on a resource will assert.
//     3. Shutdown (possibly asynchronously, if paradigm allows) the associated low-level resource.
//        Shutdown must ensure that if the state of a resource is clean, it will remain so
//        indefinitely. The state cleanness is resource-specific--for example, streams might define
//        it as "no outstanding writes and no unread data".
//     4. Check if the state of the resource is clean. If it is not, call Reset AO and return an
//        error (preferably, EBUSY).
//     5. Free (possibly asynchronously, if paradigm allows) the associated low-level resource.
//     6. Return success.
//
//  * Reset AO:
//     1. If paradigm == Paradigm::Async, schedule returning an error (preferably, ECANCELED) from
//        the current resource awaiters.
//     2. Ensure that further attempts to wait on a resource will assert.
//     3. Free synchronously the associated low-level resource. Preferably, this should be done in a
//        way that cleanly indicates an error for the event producer.
//     4. Return synchronously.
template<Paradigm paradigm>
class HybridResource {
    AK_MAKE_NONCOPYABLE(HybridResource);
    AK_MAKE_NONMOVABLE(HybridResource);

public:
    HybridResource() = default;

    // Destructor of an HybridResource must perform the following steps when called:
    // 1. If paradigm == Paradigm::Async, assert that nobody is awaiting on the resource.
    // 2. If resource is open, perform Reset AO.
    virtual ~HybridResource() = default;

    // reset() must perform the following steps when called:
    // 1. Assert that the resource is open and not in a critical section (see AsynchronousDesign.md
    //    for the definition of a critical section).
    // 2. Perform Reset AO.
    virtual void reset() = 0;

    // close() must perform the following steps when called:
    // 1. Assert that the object is fully constructed. For example, a socket might assert that it is
    //    connected.
    // 2. Assert that the resource is open and not in a critical section.
    // 3. Perform Close AO, await (if necessary) and return its result.
    virtual WrapIntoCoroutine<paradigm, ErrorOr<void>> close() = 0;

    // Resource is said to be in an error state if either Reset AO was invoked or if an operation on
    // a resource has failed and an implementation deemed the error unrecoverable. If a resource is
    // being transitioned to an error state because of an internal error, Reset AO (or its
    // equivalent) must be executed by an implementation. Resource is said to be open if it is
    // not in a error state and Close AO has never been called on it. Calling is_open in a critical
    // section asserts.
    virtual bool is_open() const = 0;
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
