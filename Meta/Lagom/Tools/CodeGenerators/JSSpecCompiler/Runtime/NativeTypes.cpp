/*
 * Copyright (c) 2024, Dan Klishch <danilklishch@gmail.com>
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include "Runtime/NativeTypes.h"
#include "Function.h"

namespace JSSpecCompiler::Runtime {

void Function::do_dump(Printer& printer) const
{
    printer.format("{}", m_declaration->name());
}

}
