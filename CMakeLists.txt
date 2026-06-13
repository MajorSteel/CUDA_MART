cmake_minimum_required(VERSION 3.18)
project(CudaMART VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Include directories
include_directories(include)

# Executable
add_executable(mart_app 
    src/main.cpp 
    src/mart_algorithm.cu
)

# Target properties
set_target_properties(mart_app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Add options for testing
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_executable(test_mart
        tests/test_mart.cpp
        src/mart_algorithm.cu
    )
    set_target_properties(test_mart PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME MartCoreTest COMMAND test_mart)
endif()
