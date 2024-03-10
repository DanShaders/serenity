/*
 * Copyright (c) 2024, Dan Klishch <danilklishch@gmail.com>
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <AK/ByteString.h>
#include <AK/HashTable.h>
#include <AK/SourceGenerator.h>
#include <AK/Vector.h>

#include "CompilationPipeline.h"

namespace JSSpecCompiler::Generator {

namespace {
class CppTypeName;
}

class Step : public CompilationStep {
public:
    Step(ByteString output_directory)
        : CompilationStep("generator"sv)
        , m_output_directory(output_directory)
    {
    }

    void run(TranslationUnitRef translation_unit) override;

private:
    struct File {
        ByteString path;
        ByteString source;
    };

    template<typename Func>
    void recurse(Runtime::Object* object, Func const& func);

    template<typename Func>
    void do_recurse(HashTable<Runtime::Object*>& visited, Runtime::Object* object, Func const& func);

    void emit_type(Runtime::ObjectType* type, Optional<Runtime::Object*> intrinsic);
    CppTypeName emit_constructor(SourceGenerator& header, Runtime::ObjectType* type, Runtime::Object* intrinsic);

    ByteString m_output_directory;
    Vector<File> m_generated_files;
    TranslationUnitRef m_translation_unit;
};

}
