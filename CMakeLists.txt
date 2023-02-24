# ########################################################################
# Copyright 2022 Advanced Micro Devices, Inc.
# ########################################################################
#Adding pthread flag for linking
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
macro(check_mpi mpi_compiler mpi_lib_a mpi_lib_so)
    find_program(MPI_MPICXX ${mpi_compiler})
    if (MPI_MPICXX)
        message ("-- ${mpi_compiler} found @ ${MPI_MPICXX}")
        if (${CMAKE_VERSION} VERSION_LESS "3.20.0")
            get_filename_component(mpi.tmpdir ${MPI_MPICXX} DIRECTORY)
            get_filename_component(mpi_base_dir ${mpi.tmpdir} DIRECTORY)
        else()
            cmake_path(GET MPI_MPICXX PARENT_PATH mpi.tmpdir)
            cmake_path(GET mpi.tmpdir PARENT_PATH mpi_base_dir)
        endif()
        find_file(MPI_H mpi.h PATHS ${mpi_base_dir} PATH_SUFFIXES include include/x86_64-linux-gnu ${ARGN} {REQUIRED)
        if (${CMAKE_VERSION} VERSION_LESS "3.20.0")
            get_filename_component(mpi_inc_dir ${MPI_H} DIRECTORY)
        else()
            cmake_path(GET MPI_H PARENT_PATH mpi_inc_dir)
        endif()
        message ("-- mpi.h is in ${mpi_inc_dir}")
        find_file(MPI_LIB NAMES ${mpi_lib_so} ${mpi_lib_a} PATHS ${mpi_base_dir} PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu REQIRED)
        message ("-- libmpi is ${MPI_LIB}")
        add_definitions(-DMPI_SUPPORT)
        include_directories(${mpi_inc_dir})
        link_libraries(${MPI_LIB})
    else()
        message ("-- ${mpi_compiler} not found")
    endif()
endmacro()

cmake_minimum_required(VERSION 3.16.3 FATAL_ERROR)

project(RCCL-tests VERSION 2.12.10 LANGUAGES CXX)

# Get ROCm path from environment if available
if (DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to ROCm installation")
else()
    set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")
endif()

# Set CMake/CPack variables
list( APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/llvm)
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Prefix install path")
set(CPACK_PACKAGING_INSTALL_PREFIX "${ROCM_PATH}" CACHE PATH "Path to install to when packaged.")
set(CMAKE_CXX_STANDARD 14)

# Get additional packages required
find_package(ROCM 0.7.3 CONFIG REQUIRED PATHS "${ROCM_PATH}")
find_package(RCCL HINTS CONFIG REQUIRED PATHS "${ROCM_PATH}")

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMCheckTargetIds)
include(ROCMClients)

# Build variables
option(NO_MPI "Build RCCL-tests without MPI support.")
## Get default GPU targets using rocm_check_target_ids
rocm_check_target_ids(
    DEFAULT_AMDGPU_TARGETS
    TARGETS "gfx803;gfx900:xnack-;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx1030"
)
set(AMDGPU_TARGETS "${DEFAULT_AMDGPU_TARGETS}" CACHE STRING "List of specific machine types for these tests to target.")

if (NOT NO_MPI)
    # Check for MPICH first
    check_mpi(mpicxx.mpich libmpich.a libmpich.so include/x86_64-linux-gnu/mpich)

    # Check for MPI in general. If we find mpicxx, we don't know whether its
    # MPICH or another MPI implementation
    if (NOT MPI_MPICXX)
        check_mpi(mpicxx libmpi.a libmpi.so)
    endif()

    if (NOT MPI_MPICXX)
        message ("-- no MPI library found")
    endif()
else()
    message ("-- MPI support explicitely disabled")
endif()

set(ROCM_USE_DEV_COMPONENT OFF)  # This repo doesn't have a dev component

# Add all of the tests
add_subdirectory(src)

# Create ROCm standard packages
rocm_create_package(
    NAME rccl-separate-tests
    DESCRIPTION "Tests for the ROCm Communication Collectives Library"
    MAINTAINER "RCCL Maintainer <rccl-maintainer@amd.com>"
)
