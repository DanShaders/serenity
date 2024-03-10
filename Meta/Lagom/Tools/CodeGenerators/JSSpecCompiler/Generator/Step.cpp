/*
 * Copyright (c) 2024, Dan Klishch <danilklishch@gmail.com>
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <AK/SourceGenerator.h>

#include "Function.h"
#include "Generator/Step.h"
#include "Runtime/NativeTypes.h"
#include "Runtime/Object.h"

namespace JSSpecCompiler::Generator {

namespace {
class CppTypeName {
    AK_MAKE_DEFAULT_COPYABLE(CppTypeName);
    AK_MAKE_DEFAULT_MOVABLE(CppTypeName);

public:
    CppTypeName(Runtime::ObjectCategory category, QualifiedName const& name)
    {
        m_name.append("JS"_string);
        for (auto const& component : name.components())
            m_name.append(component.to_string());

        if (category == Runtime::ObjectCategory::Constructor) {
            m_name.last() = MUST(String::formatted("{}Constructor", m_name.last()));
        } else {
            TODO();
        }
    }

    String cpp_namespace() const
    {
        return MUST(String::join("::"sv, m_name.span().slice(0, m_name.size() - 1)));
    }

    String unqualified_name() const
    {
        return m_name.last();
    }

    String base_file_path() const
    {
        return MUST(String::join("/"sv, m_name.span().slice(1)));
    }

private:
    Vector<String> m_name;
};
}

void Step::run(TranslationUnitRef translation_unit)
{
    m_translation_unit = translation_unit;
    auto global_object = translation_unit->realm()->global_object();

    recurse(global_object, [&](Runtime::Object* object) {
        auto type = object->type();
        if (type.has_value())
            emit_type(type.value(), object);

        auto property = object->properties().get(Runtime::WellKnownSymbol::InstanceType);
        if (property.has_value()) {
            auto type = verify_cast<Runtime::ObjectType>(property.value().get<Runtime::DataProperty>().value);
            emit_type(type, {});
        }
    });
}

template<typename Func>
void Step::recurse(Runtime::Object* object, Func const& func)
{
    HashTable<Runtime::Object*> visited;
    do_recurse(visited, object, func);
}

template<typename Func>
void Step::do_recurse(HashTable<Runtime::Object*>& visited, Runtime::Object* object, Func const& func)
{
    if (visited.contains(object))
        return;
    visited.set(object);

    func(object);

    for (auto& [key, value] : object->properties()) {
        if (value.has<Runtime::DataProperty>()) {
            auto cell = value.get<Runtime::DataProperty>().value;
            if (is<Runtime::Object>(cell))
                do_recurse(visited, dynamic_cast<Runtime::Object*>(cell), func);
        }
    }
}

void Step::emit_type(Runtime::ObjectType* type, Optional<Runtime::Object*> maybe_intrinsic)
{
    Runtime::ObjectCategory category;
    if (maybe_intrinsic.has_value()) {
        auto intrinsic = *maybe_intrinsic;
        if (!intrinsic->has(Runtime::WellKnownSymbol::ObjectCategory))
            return;

        auto enum_object = intrinsic
                               ->get(Runtime::WellKnownSymbol::ObjectCategory)
                               .as_data_property_with<Runtime::Enum<Runtime::ObjectCategory>>();
        category = enum_object->value();
    } else {
        category = Runtime::ObjectCategory::InstanceType;
    }

    StringBuilder header_source;
    SourceGenerator header { header_source };

    Optional<CppTypeName> cpp_name;

    if (category == Runtime::ObjectCategory::Constructor) {
        cpp_name = emit_constructor(header, type, maybe_intrinsic.value());
    }

    if (cpp_name.has_value()) {
        auto base_filename = ByteString::formatted("{}/{}", m_output_directory, cpp_name.value().base_file_path());
        auto header_filename = ByteString::formatted("{}.h", base_filename);

        outln("== {} ==\n{}", header_filename, header_source.string_view());
    }
}

CppTypeName Step::emit_constructor(SourceGenerator& header, Runtime::ObjectType* constructor_type, Runtime::Object* /*intrinsic*/)
{
    CppTypeName type_name(Runtime::ObjectCategory::Constructor, constructor_type->name());

    header.set("class.namespace", type_name.cpp_namespace());
    header.set("class.unqualified_name", type_name.unqualified_name());

    header.append(R"(#pragma once

#include <LibJS/Runtime/NativeFunction.h>

namespace @class.namespace@ {

class @class.unqualified_name@ final : public NativeFunction {
    JS_OBJECT(@class.unqualified_name@, NativeFunction);
    JS_DECLARE_ALLOCATOR(@class.unqualified_name@);

public:
    virtual void initialize(Realm&) override;
    virtual ~@class.unqualified_name@() override = default;

    virtual ThrowCompletionOr<Value> call() override;
    virtual ThrowCompletionOr<NonnullGCPtr<Object>> construct(FunctionObject& new_target) override;

private:
    explicit @class.unqualified_name@(Realm&);

    virtual bool has_constructor() const override { return true; }
};

}
)");

    return type_name;
}

}
