/*
 * Copyright (c) 2024, Dan Klishch <danilklishch@gmail.com>
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <AK/Enumerate.h>

#include "Runtime/Object.h"
#include "Runtime/Realm.h"

namespace JSSpecCompiler::Runtime {

Realm::Realm(DiagnosticEngine& diag)
    : m_diag(diag)
    , m_global_object(Object::create(this))
{
}

Optional<Runtime::Object*> Realm::create_object_chain(QualifiedName const& name, Location location)
{
    auto object = global_object();
    for (auto [i, component] : enumerate(name.components())) {
        auto& properties = object->properties();
        auto it = properties.find(PropertyKey { StringPropertyKey { component } });

        if (it == properties.end()) {
            object = Object::create(this);
            properties.set(
                StringPropertyKey { component },
                DataProperty {
                    .value = object,
                    .location = location,
                });
        } else {
            auto maybe_property = it->value.get_data_property_or_diagnose(this, name.slice(0, i + 1), location);
            if (!maybe_property.has_value())
                return {};

            auto maybe_object = maybe_property.value().get_or_diagnose<Runtime::Object>(this, name.slice(0, i + 1), location);
            if (!maybe_object.has_value())
                return {};

            object = maybe_object.value();
        }
    }
    return object;
}

}
