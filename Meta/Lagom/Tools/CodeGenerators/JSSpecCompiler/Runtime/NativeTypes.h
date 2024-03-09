/*
 * Copyright (c) 2024, Dan Klishch <danilklishch@gmail.com>
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include "Runtime/Realm.h"

namespace JSSpecCompiler::Runtime {

template<typename T>
class Enum : public Cell {
public:
    static constexpr StringView TYPE_NAME = "native enumeration"sv;

    static Enum<T>* create(Realm* realm, T value)
    {
        return realm->adopt_cell(new Enum { value });
    }

    StringView type_name() const override { return TYPE_NAME; }

    T value() const { return m_value; }

protected:
    void do_dump(Printer& printer) const override
    {
        printer.format("{}", to_string(m_value));
    }

private:
    Enum(T value)
        : m_value(value)
    {
    }

    T m_value;
};

class Function : public Cell {
public:
    static constexpr StringView TYPE_NAME = "native function"sv;

    static Function* create(Realm* realm, FunctionDeclarationRef declaration)
    {
        return realm->adopt_cell(new Function { declaration });
    }

    StringView type_name() const override { return TYPE_NAME; }

    FunctionDeclarationRef declaration() const { return m_declaration; }

protected:
    void do_dump(Printer& printer) const override;

private:
    Function(FunctionDeclarationRef declaration)
        : m_declaration(declaration)
    {
    }

    FunctionDeclarationRef m_declaration;
};

}
