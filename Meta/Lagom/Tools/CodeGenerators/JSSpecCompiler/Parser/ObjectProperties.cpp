/*
 * Copyright (c) 2023-2024, Dan Klishch <danilklishch@gmail.com>
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "Parser/Lexer.h"
#include "Parser/SpecificationParsing.h"
#include "Parser/XMLUtils.h"
#include "Runtime/NativeTypes.h"
#include "Runtime/Object.h"

namespace JSSpecCompiler {

bool ObjectProperties::post_initialize(XML::Node const* element)
{
    auto* realm = context().translation_unit()->realm();
    auto location = context().location_from_xml_offset(element->offset);
    auto const& header = m_header.header.get<ClauseHeader::PropertiesList>();

    if (header.object_type == ClauseHeader::ObjectType::Instance) {
        auto maybe_object = realm->create_object_chain(header.name, location);
        if (!maybe_object.has_value())
            return false;
        auto object = *maybe_object;

        auto instance_type = Runtime::ObjectType::create(realm);

        Runtime::PropertyKey key = Runtime::WellKnownSymbol::InstanceType;
        Runtime::Property value = Runtime::DataProperty {
            .value = instance_type,
            .location = location,
        };

        if (object->has(key)) {
            context().diag().error(location,
                "instance type is redefined");
            context().diag().note(object->get(key).location(),
                "previously defined here");
            return false;
        } else {
            object->set(key, move(value));
        }
    } else {
        auto name = header.name;

        if (header.object_type == ClauseHeader::ObjectType::Prototype)
            name = name.with_appended("prototype"_fly_string);

        auto init_or_check_intrinsic = [&](Runtime::Object* object) -> bool {
            Runtime::ObjectCategory category;
            if (header.object_type == ClauseHeader::ObjectType::Prototype)
                category = Runtime::ObjectCategory::Prototype;
            else if (header.object_type == ClauseHeader::ObjectType::Constructor)
                category = Runtime::ObjectCategory::Constructor;
            else
                VERIFY_NOT_REACHED();

            if (object->has(Runtime::WellKnownSymbol::ObjectCategory)) {
                auto& property = object->get(Runtime::WellKnownSymbol::ObjectCategory);
                auto old_category = property.as_data_property_with<Runtime::Enum<Runtime::ObjectCategory>>()->value();
                if (old_category != category) {
                    context().diag().error(location,
                        "object category is redefined to {}", to_string(category));
                    context().diag().note(property.location(),
                        "previously defined here as {}", to_string(old_category));
                    return false;
                }
            } else {
                object->set(Runtime::WellKnownSymbol::ObjectCategory,
                    Runtime::DataProperty {
                        .value = Runtime::Enum<Runtime::ObjectCategory>::create(realm, category),
                        .location = location,
                    });
            }
            return true;
        };

        auto maybe_parent = realm->create_object_chain(name.without_last_component(), location);
        if (!maybe_parent.has_value())
            return false;
        auto parent = *maybe_parent;

        Runtime::Object* object;

        Runtime::PropertyKey key = Runtime::StringPropertyKey { name.last_component() };
        if (!parent->has(key)) {
            object = Runtime::Object::create(realm);
            parent->set(key, Runtime::DataProperty {
                                 .value = object,
                                 .location = location,
                             });
            if (!init_or_check_intrinsic(object))
                return false;
        } else {
            auto maybe_property = parent->get(key).get_data_property_or_diagnose(realm, name, location);
            if (!maybe_property.has_value())
                return false;
            auto property = *maybe_property;

            auto maybe_object = property.get_or_diagnose<Runtime::Object>(realm, name, location);
            if (!maybe_object.has_value())
                return false;
            object = *maybe_object;

            if (!init_or_check_intrinsic(object))
                return false;
        }

        m_enclosing_type = object->type().value();
    }

    return true;
}

void ObjectProperties::do_collect(TranslationUnitRef /*translation_unit*/)
{
    m_saved_enclosing_type = context().enclosing_type();
    context().set_enclosing_type(m_enclosing_type);
}

void ObjectProperties::leave()
{
    context().set_enclosing_type(m_saved_enclosing_type);
}

}
