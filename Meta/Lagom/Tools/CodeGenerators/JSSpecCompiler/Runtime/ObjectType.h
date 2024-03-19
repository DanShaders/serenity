/*
 * Copyright (c) 2024, Dan Klishch <danilklishch@gmail.com>
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include "Runtime/Realm.h"

namespace JSSpecCompiler::Runtime {

class ObjectType : public Cell {
public:
    struct AssignedFunction {
        FunctionDeclarationRef function;
        bool is_abstract_operation;
    };

    static constexpr StringView TYPE_NAME = "type"sv;

    static ObjectType* create(Realm* realm, QualifiedName&& name)
    {
        return realm->adopt_cell(new ObjectType { move(name) });
    }

    StringView type_name() const override { return TYPE_NAME; }

    auto& assigned_functions() { return m_assigned_functions; }

    QualifiedName& name() { return m_name; }

protected:
    void do_dump(Printer& printer) const override;

private:
    ObjectType(QualifiedName&& name)
        : m_name(move(name))
    {
    }

    QualifiedName m_name;
    Vector<AssignedFunction> m_assigned_functions;
};

}
