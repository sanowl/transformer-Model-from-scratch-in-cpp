cmake_minimum_required(VERSION 3.15)
project(TransformerModel CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra")
    
    # Enable AVX2 for SIMD operations
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /arch:AVX2")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi /W4")
endif()

# Find OpenMP for parallel processing
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP found - enabling parallel processing")
else()
    message(WARNING "OpenMP not found - parallel processing disabled")
endif()

# Source files
set(SOURCES
    matrix.cpp
    tensor.cpp
    attention.cpp
    transformer.cpp
    trainer.cpp
    main.cpp
)

set(HEADERS
    matrix.h
    tensor.h
    attention.h
    transformer.h
    trainer.h
)

# Create the executable
add_executable(transformer ${SOURCES} ${HEADERS})

# Link libraries
if(OpenMP_CXX_FOUND)
    target_link_libraries(transformer OpenMP::OpenMP_CXX)
endif()

# Platform-specific libraries
if(WIN32)
    # Windows-specific libraries if needed
elseif(APPLE)
    # macOS-specific libraries if needed
    target_link_libraries(transformer "-framework Accelerate")
else()
    # Linux-specific libraries
    target_link_libraries(transformer m)
endif()

# Add compiler definitions
target_compile_definitions(transformer PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "C++ flags (Release): ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "C++ flags (Debug): ${CMAKE_CXX_FLAGS_DEBUG}")

# Optional: Create a test executable for component testing
option(BUILD_TESTS "Build test executable" OFF)

if(BUILD_TESTS)
    set(TEST_SOURCES
        matrix.cpp
        tensor.cpp
        attention.cpp
        test_components.cpp
    )
    
    add_executable(test_transformer ${TEST_SOURCES})
    
    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_transformer OpenMP::OpenMP_CXX)
    endif()
    
    if(APPLE)
        target_link_libraries(test_transformer "-framework Accelerate")
    elseif(NOT WIN32)
        target_link_libraries(test_transformer m)
    endif()
endif()

# Installation rules
install(TARGETS transformer DESTINATION bin)

# Package configuration
set(CPACK_PACKAGE_NAME "TransformerModel")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Custom Transformer Model Implementation in C++")
include(CPack)