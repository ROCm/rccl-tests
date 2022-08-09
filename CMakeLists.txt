# ########################################################################
# Copyright 2022 Advanced Micro Devices, Inc.
# ########################################################################

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
option(USE_MPI "Build RCCL-tests with MPI support. Requires the MPI path to be set.")
set(MPI_PATH "" CACHE PATH "Path to MPI installation")
## Get default GPU targets using rocm_check_target_ids
rocm_check_target_ids(
    DEFAULT_AMDGPU_TARGETS
    TARGETS "gfx803;gfx900:xnack-;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx1030"
)
set(AMDGPU_TARGETS "${DEFAULT_AMDGPU_TARGETS}" CACHE STRING "List of specific machine types for these tests to target.")

# Find the MPI package if we're using MPI
if (USE_MPI)
    if(NOT MPI_PATH STREQUAL "")
        set(MPI_HOME "${MPI_PATH}")
    endif()
    find_package(MPI REQUIRED MODULE)
    add_definitions(-DOMPI_SKIP_MPICXX -DMPI_SUPPORT)
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
