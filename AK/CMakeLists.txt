set(AK_SOURCES
    Assertions.cpp
    Base64.cpp
    CircularBuffer.cpp
    ConstrainedStream.cpp
    CountingStream.cpp
    DOSPackedTime.cpp
    DeprecatedFlyString.cpp
    DeprecatedString.cpp
    Error.cpp
    FloatingPointStringConversions.cpp
    FlyString.cpp
    Format.cpp
    FuzzyMatch.cpp
    GenericLexer.cpp
    Hex.cpp
    JsonObject.cpp
    JsonParser.cpp
    JsonPath.cpp
    JsonValue.cpp
    LexicalPath.cpp
    MemoryStream.cpp
    NumberFormat.cpp
    OptionParser.cpp
    Random.cpp
    StackInfo.cpp
    Stream.cpp
    String.cpp
    StringBuilder.cpp
    StringFloatingPointConversions.cpp
    StringImpl.cpp
    StringUtils.cpp
    StringView.cpp
    Time.cpp
    URL.cpp
    URLParser.cpp
    UUID.cpp
    Utf16View.cpp
    Utf32View.cpp
    Utf8View.cpp
    kmalloc.cpp
)

configure_file(Debug.h.in Debug.h)

add_library(AK_PublicInterface INTERFACE)
target_compile_options(AK_PublicInterface INTERFACE
    -Wno-literal-suffix
    -Wno-dangling-reference
    -Wno-maybe-uninitialized
)
target_include_directories(AK_PublicInterface INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    "${CMAKE_CURRENT_BINARY_DIR}/.."
)

add_library(AK_PrivateInterface INTERFACE)
target_compile_options(AK_PrivateInterface INTERFACE
    -O2 -Wall -Wextra -Wno-shadow -Werror
)

add_library(AK_Host STATIC EXCLUDE_FROM_ALL ${AK_SOURCES})
target_compile_options(AK_Host PRIVATE ${ADELAIDE_HOST_COMPILE_OPTIONS})
target_link_libraries(AK_Host PUBLIC AK_PublicInterface PRIVATE AK_PrivateInterface)
