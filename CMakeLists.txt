# ########################################################################
# Copyright 2022 Advanced Micro Devices, Inc.
# ########################################################################
#Adding pthread flag for linking
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

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
include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMCheckTargetIds)
include(ROCMClients)

# Build variables
option(USE_MPI "Build RCCL-tests with MPI support.")

# Default GPU architectures to build
#==================================================================================================
set(DEFAULT_GPUS
      gfx803
      gfx900:xnack-
      gfx906:xnack-
      gfx908:xnack-
      gfx90a:xnack-
      gfx90a:xnack+
      gfx940
      gfx941
      gfx942
      gfx1030
      gfx1100
      gfx1101
      gfx1102)

set(AMDGPU_TARGETS ${DEFAULT_GPUS} CACHE STRING "Target default GPUs if AMDGPU_TARGETS is not defined.")
## Determine which GPU architectures to build for
if (COMMAND rocm_check_target_ids)
    message(STATUS "Checking for ROCm support for GPU targets:")
    rocm_check_target_ids(SUPPORTED_GPUS TARGETS "${AMDGPU_TARGETS}")
else()
    message(WARNING "Unable to check for supported GPU targets. Falling back to default GPUs")
    set(SUPPORTED_GPUS ${DEFAULT_GPUS})
endif()
set(GPU_TARGETS "${SUPPORTED_GPUS}" CACHE STRING "List of specific GPU architectures to build for.")
message(STATUS "Compiling for ${GPU_TARGETS}")

find_package(RCCL HINTS CONFIG REQUIRED PATHS "${ROCM_PATH}")
if (RCCL_FOUND)
    message(STATUS "RCCL version : ${RCCL_VERSION}")
    message(STATUS "RCCL include path : ${RCCL_INCLUDE_DIRS}")
    message(STATUS "RCCL libraries : ${RCCL_LIBRARIES}")
endif()

if (USE_MPI)
    find_package(MPI REQUIRED)
    if (MPI_FOUND)
        message(STATUS "MPI include path : ${MPI_CXX_INCLUDE_PATH}")
        message(STATUS "MPI libraries : ${MPI_CXX_LIBRARIES}")
        add_definitions(-DMPI_SUPPORT)
    else()
        message ("-- no MPI library found")
    endif()
else()
    message ("-- MPI support disabled")
endif()

set(ROCM_USE_DEV_COMPONENT OFF)  # This repo doesn't have a dev component

# Add all of the tests
add_subdirectory(src)

# Create ROCm standard packages
rocm_create_package(
    NAME rccl-tests
    DESCRIPTION "Tests for the ROCm Communication Collectives Library"
    MAINTAINER "RCCL Maintainer <rccl-maintainer@amd.com>"
)
