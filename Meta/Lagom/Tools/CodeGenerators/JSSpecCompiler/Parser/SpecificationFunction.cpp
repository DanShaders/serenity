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
#include "Runtime/Realm.h"

namespace JSSpecCompiler {

bool SpecificationFunction::post_initialize(XML::Node const* element)
{
    VERIFY(element->as_element().name == tag_emu_clause);

    auto& ctx = context();
    m_location = ctx.location_from_xml_offset(element->offset);

    auto maybe_id = get_attribute_by_name(element, attribute_id);
    if (!maybe_id.has_value()) {
        ctx.diag().error(m_location,
            "no id attribute");
    } else {
        m_id = maybe_id.value();
    }

    m_header.header.visit(
        [&](AbstractOperationDeclaration const& abstract_operation) {
            m_declaration = abstract_operation;

            auto abstract_operation_id = get_attribute_by_name(element, attribute_aoid).value();

            if (abstract_operation.name != abstract_operation_id) {
                ctx.diag().warn(m_location,
                    "function name in header and <emu-clause>[aoid] do not match");
            }
        },
        [&](OneOf<AccessorDeclaration, MethodDeclaration> auto const& declaration) {
            m_declaration = declaration;
        },
        [&](auto const&) {
            VERIFY_NOT_REACHED();
        });

    Vector<XML::Node const*> algorithm_nodes;

    for (auto const& child : element->as_element().children) {
        child->content.visit(
            [&](XML::Node::Element const& element) {
                if (element.name == tag_h1) {
                    // Processed in SpecificationClause
                } else if (element.name == tag_p) {
                    ctx.diag().warn(ctx.location_from_xml_offset(child->offset),
                        "prose is ignored");
                } else if (element.name == tag_emu_alg) {
                    algorithm_nodes.append(child);
                } else {
                    ctx.diag().error(ctx.location_from_xml_offset(child->offset),
                        "<{}> should not be a child of <emu-clause> specifing function"sv, element.name);
                }
            },
            [&](auto const&) {});
    }

    if (algorithm_nodes.size() != 1) {
        ctx.diag().error(m_location,
            "<emu-clause> specifing function should have exactly one <emu-alg> child"sv);
        return false;
    }

    auto maybe_algorithm = Algorithm::create(ctx, algorithm_nodes[0]);
    if (maybe_algorithm.has_value()) {
        m_algorithm = maybe_algorithm.release_value();
        return true;
    } else {
        return false;
    }
}

void SpecificationFunction::do_collect(TranslationUnitRef translation_unit)
{
    auto definition = make_ref_counted<FunctionDefinition>(m_declaration.release_value(), m_location, m_algorithm.tree());
    FunctionDeclarationRef declaration = definition;
    translation_unit->adopt_function(move(definition));

    auto realm = context().translation_unit()->realm();

    m_header.header.visit(
        [&](OneOf<AccessorDeclaration, MethodDeclaration> auto const& accessor_or_method) {
            auto maybe_object = realm->create_object_chain(accessor_or_method.name.without_last_component(), m_location);
            if (!maybe_object.has_value())
                return;
            auto object = *maybe_object;

            Runtime::PropertyKey key = Runtime::StringPropertyKey { accessor_or_method.name.last_component() };
            auto& properties = object->properties();

            if (object->has(key)) {
                realm->diag().error(m_location,
                    "property {} is redefined", accessor_or_method.name.to_string());
                realm->diag().note(object->get(key).location(),
                    "previously defined here");
                return;
            }

            HashSetResult result;
            if constexpr (SameAs<decltype(accessor_or_method), AccessorDeclaration const&>) {
                result = properties.set(key,
                    Runtime::AccessorProperty {
                        .getter = declaration,
                        .setter = {},
                        .location = m_location,
                    });
            } else {
                result = properties.set(key,
                    Runtime::DataProperty {
                        .value = Runtime::Function::create(realm, declaration),
                        .location = m_location,
                    });
            }
            VERIFY(result == HashSetResult::InsertedNewEntry);
        },
        [&](auto const&) {});
}

}
