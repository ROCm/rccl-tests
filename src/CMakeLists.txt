# ########################################################################
# Copyright 2022 Advanced Micro Devices, Inc.
# ########################################################################

# Compile common object library
set_property(SOURCE common.cu timer.cc ../verifiable/verifiable.cu PROPERTY LANGUAGE CXX)
add_library(rccl_common OBJECT common.cu timer.cc ../verifiable/verifiable.cu)
if(USE_MPI)
    target_link_libraries(rccl_common roc::rccl MPI::MPI_CXX hip::device)
else()
    target_link_libraries(rccl_common roc::rccl hip::device)
endif()

function(add_relative_test test_name test_target)
    get_target_property(EXE_PATH ${test_target} RUNTIME_OUTPUT_DIRECTORY)
    if(EXE_PATH STREQUAL "EXE_PATH-NOTFOUND")
        set(EXE_PATH ".")
    endif()
    get_filename_component(EXE_PATH "${EXE_PATH}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    get_target_property(EXE_NAME ${test_target} RUNTIME_OUTPUT_NAME)
    if(EXE_NAME STREQUAL "EXE_NAME-NOTFOUND")
        get_target_property(EXE_NAME ${test_target} OUTPUT_NAME)
        if(EXE_NAME STREQUAL "EXE_NAME-NOTFOUND")
            set(EXE_NAME "${test_target}")
        endif()
    endif()
    file(RELATIVE_PATH rel_path "${CMAKE_CURRENT_BINARY_DIR}" "${EXE_PATH}/${EXE_NAME}")
    add_test(NAME "${test_name}" COMMAND "./${rel_path}")
endfunction()

function(add_rccl_test TEST)
    set(TEST_SOURCE "${TEST}.cu")
    set_property(SOURCE ${TEST_SOURCE} PROPERTY LANGUAGE CXX)
    set(TEST_TARGET "${TEST}_perf")
    add_executable(${TEST_TARGET} ${TEST_SOURCE})
    target_link_libraries(
        ${TEST_TARGET}
        PRIVATE
            rccl_common
    )
    if (NOT WIN32)
        foreach(amdgpu_target ${AMDGPU_TARGETS})
            target_link_libraries(${TEST_TARGET} PRIVATE --amdgpu-target=${amdgpu_target})
        endforeach()
    endif()
    set_target_properties(
        ${TEST_TARGET}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
            # LINKER_LANGUAGE CXX
    )
    add_relative_test(${TEST} ${TEST_TARGET})
    rocm_install(TARGETS ${TEST_TARGET})
    # TODO: copy/install DLLs on Windows
    set_target_properties(
        ${TEST_TARGET} PROPERTIES
        INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${ROCM_PATH}/lib"
    )
endfunction()

add_rccl_test(all_gather)
add_rccl_test(all_reduce)
add_rccl_test(alltoall)
add_rccl_test(alltoallv)
add_rccl_test(broadcast)
add_rccl_test(gather)
add_rccl_test(hypercube)
add_rccl_test(reduce_scatter)
add_rccl_test(reduce)
add_rccl_test(scatter)
add_rccl_test(sendrecv)
