cmake_minimum_required(VERSION 3.18)
project(cutlass_blackwell_gemm CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 100)

# Find CUTLASS
set(CUTLASS_DIR "${CMAKE_SOURCE_DIR}/../../../third_party/cutlass")
if(NOT EXISTS ${CUTLASS_DIR})
    message(FATAL_ERROR "CUTLASS not found at ${CUTLASS_DIR}")
endif()

# Find Python and PyTorch
find_package(Python REQUIRED COMPONENTS Interpreter Development)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX})
find_package(Torch REQUIRED)

# Get pybind11 include
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cpp_extension.include_paths()[0])"
    OUTPUT_VARIABLE TORCH_INCLUDE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Include paths
include_directories(
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
    ${Python_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_INCLUDE}
    ${TORCH_INCLUDE}/torch/csrc/api/include
)

# Compile the CUTLASS-based GEMM kernel as Python module
Python_add_library(cutlass_blackwell_gemm MODULE
    cutlass_gemm.cu
)

target_compile_options(cutlass_blackwell_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --expt-relaxed-constexpr
        -gencode=arch=compute_100a,code=sm_100a
        -gencode=arch=compute_103a,code=sm_103a
        -D_GLIBCXX_USE_CXX11_ABI=0
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -D_GLIBCXX_USE_CXX11_ABI=0
    >
)

target_link_libraries(cutlass_blackwell_gemm PRIVATE
    ${TORCH_LIBRARIES}
    Python::Python
    cuda
)

set_target_properties(cutlass_blackwell_gemm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    PREFIX ""
    SUFFIX ".so"
)

