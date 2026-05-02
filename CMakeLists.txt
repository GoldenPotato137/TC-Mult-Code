cmake_minimum_required(VERSION 3.19)

project(TC_Mult_Code LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;89" CACHE STRING "CUDA architectures for TC-Mult-Code" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

set(TC_MULT_WORDS_A "192" CACHE STRING "32-bit limb count for operand A")
set(TC_MULT_WORDS_B "${TC_MULT_WORDS_A}" CACHE STRING "32-bit limb count for operand B")
option(TC_MULT_BUILD_PRIVATE_IMPL "Build the private TC-Mult binary from sources outside this review package" OFF)
set(TC_MULT_PRIVATE_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." CACHE PATH "bullet_proof_gpu source root used only when TC_MULT_BUILD_PRIVATE_IMPL=ON")

if(NOT TC_MULT_WORDS_A STREQUAL TC_MULT_WORDS_B)
    message(FATAL_ERROR "TC-Mult package currently expects TC_MULT_WORDS_A == TC_MULT_WORDS_B")
endif()

set(TC_MULT_PUBLIC_DEFS
    TC_MULT_WORDS_A=${TC_MULT_WORDS_A}
    TC_MULT_WORDS_B=${TC_MULT_WORDS_B}
    BP_BIGNUM_WORDS_A=${TC_MULT_WORDS_A}
    BP_BIGNUM_WORDS_B=${TC_MULT_WORDS_B}
)

add_library(tc_mult_base STATIC
    src/base/tc_mult_base.cu
)
target_include_directories(tc_mult_base
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_definitions(tc_mult_base PUBLIC ${TC_MULT_PUBLIC_DEFS})
target_link_libraries(tc_mult_base PUBLIC CUDA::cudart)
set_target_properties(tc_mult_base PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(TC_MULT_IMPL_LIB_BASENAME "tc_mult_w${TC_MULT_WORDS_A}_sm80_sm89")
set(TC_MULT_IMPL_LIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/lib/lib${TC_MULT_IMPL_LIB_BASENAME}.a")
set(TC_MULT_IMPL_AVAILABLE OFF)

if(TC_MULT_BUILD_PRIVATE_IMPL)
    set(TC_MULT_PRIVATE_BIGNUM_DIR "${TC_MULT_PRIVATE_SOURCE_ROOT}/test/tests/bignum_mult")
    string(CONCAT TC_MULT_PRIVATE_CORE_SOURCE "ten" "sor" "_v" "2")
    set(TC_MULT_PRIVATE_INPUT_SOURCES
        "${TC_MULT_PRIVATE_CORE_SOURCE}.cu"
        "${TC_MULT_PRIVATE_CORE_SOURCE}_gmem_b.cu"
        "${TC_MULT_PRIVATE_CORE_SOURCE}_smem_carry.cu"
    )
    set(TC_MULT_PRIVATE_OUTPUT_SOURCES
        "tc_mult_impl.cu"
        "tc_mult_impl_gmem_b.cu"
        "tc_mult_impl_wo_perm.cu"
    )

    foreach(required_source IN LISTS TC_MULT_PRIVATE_INPUT_SOURCES)
        if(NOT EXISTS "${TC_MULT_PRIVATE_BIGNUM_DIR}/${required_source}")
            message(FATAL_ERROR "Private TC-Mult source not found: ${required_source}")
        endif()
    endforeach()

    set(TC_MULT_GENERATED_PRIVATE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated_private")
    file(MAKE_DIRECTORY "${TC_MULT_GENERATED_PRIVATE_DIR}")
    set(TC_MULT_PRIVATE_IMPL_SOURCES)
    list(LENGTH TC_MULT_PRIVATE_INPUT_SOURCES TC_MULT_PRIVATE_SOURCE_COUNT)
    math(EXPR TC_MULT_PRIVATE_SOURCE_LAST "${TC_MULT_PRIVATE_SOURCE_COUNT} - 1")
    foreach(source_index RANGE ${TC_MULT_PRIVATE_SOURCE_LAST})
        list(GET TC_MULT_PRIVATE_INPUT_SOURCES ${source_index} input_source)
        list(GET TC_MULT_PRIVATE_OUTPUT_SOURCES ${source_index} output_source)
        file(READ "${TC_MULT_PRIVATE_BIGNUM_DIR}/${input_source}" source_text)
        string(REPLACE "#include \"../../test.hpp\"" "#include \"tc_mult_private_test.hpp\"" source_text "${source_text}")
        file(WRITE "${TC_MULT_GENERATED_PRIVATE_DIR}/${output_source}" "${source_text}")
        list(APPEND TC_MULT_PRIVATE_IMPL_SOURCES "${TC_MULT_GENERATED_PRIVATE_DIR}/${output_source}")
    endforeach()

    add_library(tc_mult_bin STATIC
        src/private/tc_mult_api.cu
        ${TC_MULT_PRIVATE_IMPL_SOURCES}
    )
    target_include_directories(tc_mult_bin
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src/private
            ${TC_MULT_PRIVATE_BIGNUM_DIR}
            ${TC_MULT_PRIVATE_SOURCE_ROOT}/test
            ${TC_MULT_PRIVATE_SOURCE_ROOT}/include
            ${TC_MULT_PRIVATE_SOURCE_ROOT}/depends
            ${TC_MULT_PRIVATE_SOURCE_ROOT}/depends/cutlass/include
            ${TC_MULT_PRIVATE_SOURCE_ROOT}/depends/sppark/depends/blst/src
    )
    target_compile_definitions(tc_mult_bin PUBLIC ${TC_MULT_PUBLIC_DEFS})
    target_compile_options(tc_mult_bin PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=23>
        $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=815>
        $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=177>
        $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=20054>
    )
    target_link_libraries(tc_mult_bin PUBLIC CUDA::cudart)
    set_target_properties(tc_mult_bin PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
        OUTPUT_NAME ${TC_MULT_IMPL_LIB_BASENAME}
    )
    set(TC_MULT_IMPL_AVAILABLE ON)
elseif(EXISTS "${TC_MULT_IMPL_LIB_PATH}")
    add_library(tc_mult_bin STATIC IMPORTED GLOBAL)
    set_target_properties(tc_mult_bin PROPERTIES
        IMPORTED_LOCATION "${TC_MULT_IMPL_LIB_PATH}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include"
        INTERFACE_COMPILE_DEFINITIONS "TC_MULT_WORDS_A=${TC_MULT_WORDS_A};TC_MULT_WORDS_B=${TC_MULT_WORDS_B}"
        INTERFACE_LINK_LIBRARIES CUDA::cudart
    )
    set(TC_MULT_IMPL_AVAILABLE ON)
else()
    message(WARNING "TC-Mult binary not found at ${TC_MULT_IMPL_LIB_PATH}. Configure with -DTC_MULT_BUILD_PRIVATE_IMPL=ON to generate it, or copy the review binary into lib/.")
endif()

function(add_tc_mult_example target source)
    add_executable(${target} ${source})
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples)
    target_compile_definitions(${target} PRIVATE ${TC_MULT_PUBLIC_DEFS})
    target_link_libraries(${target} PRIVATE tc_mult_base CUDA::cudart)
    set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endfunction()

add_tc_mult_example(example_base examples/example_base.cu)

set(TC_MULT_BENCHMARK_BASELINE_LIBS)
set(TC_MULT_ALL_BASELINES_AVAILABLE ON)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cgbn/include/cgbn/cgbn.h")
    add_library(tc_mult_cgbn STATIC src/baselines/tc_mult_cgbn.cu)
    target_include_directories(tc_mult_cgbn
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cgbn/include
    )
    target_compile_definitions(tc_mult_cgbn PUBLIC ${TC_MULT_PUBLIC_DEFS})
    target_link_libraries(tc_mult_cgbn PUBLIC CUDA::cudart gmp)
    set_target_properties(tc_mult_cgbn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    list(APPEND TC_MULT_BENCHMARK_BASELINE_LIBS tc_mult_cgbn)
    message(STATUS "CGBN baseline submodule detected")
else()
    set(TC_MULT_ALL_BASELINES_AVAILABLE OFF)
    message(STATUS "CGBN baseline submodule not initialized; run git submodule update --init --recursive")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/GPU-NTT/src/include/gpuntt/ntt_merge/ntt.cuh")
    add_library(tc_mult_gpuntt STATIC
        src/baselines/tc_mult_gpuntt.cu
        third_party/GPU-NTT/src/lib/common/common.cu
        third_party/GPU-NTT/src/lib/common/nttparameters.cu
        third_party/GPU-NTT/src/lib/ntt_merge/ntt.cu
    )
    target_include_directories(tc_mult_gpuntt
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/GPU-NTT/src/include
    )
    target_compile_definitions(tc_mult_gpuntt PUBLIC ${TC_MULT_PUBLIC_DEFS})
    target_compile_options(tc_mult_gpuntt PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
    target_link_libraries(tc_mult_gpuntt PUBLIC CUDA::cudart)
    set_target_properties(tc_mult_gpuntt PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    list(APPEND TC_MULT_BENCHMARK_BASELINE_LIBS tc_mult_gpuntt)
    message(STATUS "GPU-NTT baseline submodule detected")
else()
    set(TC_MULT_ALL_BASELINES_AVAILABLE OFF)
    message(STATUS "GPU-NTT baseline submodule not initialized; run git submodule update --init --recursive")
endif()

if(TC_MULT_IMPL_AVAILABLE)
    foreach(target example_tc_mult verify_all)
        add_executable(${target} examples/${target}.cu)
        target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples)
        target_compile_definitions(${target} PRIVATE ${TC_MULT_PUBLIC_DEFS})
        target_link_libraries(${target} PRIVATE tc_mult_base tc_mult_bin CUDA::cudart)
        set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endforeach()

    if(TC_MULT_ALL_BASELINES_AVAILABLE)
        add_executable(benchmark examples/benchmark.cu)
        target_include_directories(benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples)
        target_compile_definitions(benchmark PRIVATE ${TC_MULT_PUBLIC_DEFS})
        target_link_libraries(benchmark PRIVATE tc_mult_base tc_mult_bin ${TC_MULT_BENCHMARK_BASELINE_LIBS} CUDA::cudart)
        set_target_properties(benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    else()
        message(STATUS "benchmark target requires CGBN and GPU-NTT baseline submodules")
    endif()
endif()
