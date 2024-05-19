set(SOURCES
<<<<<<< HEAD
=======
    AsyncDeflate.cpp
    AsyncZlib.cpp
>>>>>>> 2f9ab3d685 (LibCompress: Add asynchronous zlib decompressor)
    Brotli.cpp
    BrotliDictionary.cpp
    Deflate.cpp
    Lzma.cpp
    Lzma2.cpp
    PackBitsDecoder.cpp
    Xz.cpp
    Zlib.cpp
    Gzip.cpp
)

serenity_lib(LibCompress compress)
target_link_libraries(LibCompress PRIVATE LibCore LibCrypto)
