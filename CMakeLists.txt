# ########################################################################
# Copyright 2022 Advanced Micro Devices, Inc.
# ########################################################################
#Adding pthread flag for linking
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
macro(check_mpi mpi_compiler mpi_lib_a mpi_lib_so mpi_bin_dir mpi_base_lib_dir mpi_inc_dir)
    find_program(MPI_MPICXX ${mpi_compiler} PATHS ${mpi_bin_dir} NO_DEFAULT_PATH)
    if (MPI_MPICXX)
        message ("-- ${mpi_compiler} found @ ${MPI_MPICXX}")
        find_file(MPI_H mpi.h PATHS ${mpi_inc_dir} NO_DEFAULT_PATH)
        message ("-- mpi.h is in ${MPI_H}")
        find_file(MPI_LIB NAMES ${mpi_lib_so} ${mpi_lib_a} PATHS ${mpi_base_lib_dir} PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu NO_DEFAULT_PATH)
        message ("-- libmpi is ${MPI_LIB}")
	if (NOT MPI_H OR NOT MPI_LIB)
	    set (MPI_MPICXX "MPI_MPICXX-NOTFOUND")
	    set (MPI_H "MPI_H-NOTFOUND")
	    set (MPI_LIB "MPI_LIB-NOTFOUND")
	else()
            add_definitions(-DMPI_SUPPORT)
            include_directories(${mpi_inc_dir})
            link_libraries(${MPI_LIB})
	endif()
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
option(MPI_PATH "Use MPI in the specified directory.")
## Get default GPU targets using rocm_check_target_ids
rocm_check_target_ids(
    DEFAULT_AMDGPU_TARGETS
    TARGETS "gfx803;gfx900:xnack-;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx1030"
)
set(AMDGPU_TARGETS "${DEFAULT_AMDGPU_TARGETS}" CACHE STRING "List of specific machine types for these tests to target.")

if (NOT NO_MPI)
    # CHECK for MPI Path first. User requested this directory explicitely
    if (MPI_PATH)
        set(mpi_spec_bin_dir "${MPI_PATH}/bin")
	set(mpi_spec_inc_dir "${MPI_PATH}/include")
        check_mpi(mpicxx libmpi.a libmpi.so ${mpi_spec_bin_dir} ${MPI_PATH} ${mpi_spec_inc_dir})
	if (NOT MPI_MPICXX)
            # Since the user explicitely requested this directory, abort if something went wrong.
	    MESSAGE(FATAL_ERROR "Could not find MPI in ${MPI_PATH}")
        endif()
    endif()

    # Check for MPICH Ubuntu installation
    if (NOT MPI_MPICXX)
        check_mpi(mpicxx.mpich libmpich.a libmpich.so /usr/bin /usr /usr/include/x86_64-linux-gnu/mpich)
    endif()

    # Check for Open MPI Ubuntu installation
    if (NOT MPI_MPICXX)
        check_mpi(mpicxx.openmpi libmpi.a libmpi.so /usr/bin  /usr/lib/x86_64-linux-gnu/openmpi /usr/lib/x86_64-linux-gnu/openmpi/include)
    endif()

    # Check for MPICH RHEL installation
    if (NOT MPI_MPICXX)
        check_mpi(mpicxx libmpich.a libmpich.so /usr/lib64/mpich/bin /usr/lib64/mpich /usr/include/mpich-x86_64)
    endif()

    # Check for Open MPI RHEL installation
    if (NOT MPI_MPICXX)
        check_mpi(mpicxx libmpi.a libmpi.so /usr/lib64/openmpi/bin /usr/lib64/openmpi /usr/include/openmpi-x64_64)
    endif()

    # Check for MPICH SLES installation
    if (NOT MPI_MPICXX)
        check_mpi(mpicxx libmpich.a libmpich.so /usr/lib64/mpi/gcc/mpich/bin /usr/lib64/mpi/gcc/mpich /usr/lib64/mpi/gcc/mpich/include)
    endif()

    # Check for Open MPI v4 SLES installation
    if (NOT MPI_MPICXX)
        check_mpi(mpicxx libmpi.a libmpi.so /usr/lib64/mpi/gcc/openmpi4/bin /usr/lib64/mpi/gcc/openmpi4 /usr/lib64/mpi/gcc/openmpi4/include)
    endif()

    # Check for Open MPI v3 SLES installation
    if (NOT MPI_MPICXX)
        check_mpi(mpicxx libmpi.a libmpi.so /usr/lib64/mpi/gcc/openmpi3/bin /usr/lib64/mpi/gcc/openmpi3 /usr/lib64/mpi/gcc/openmpi3/include)
    endif()
    
    # Check for Open MPI v2 SLES installation
    if (NOT MPI_MPICXX)
        check_mpi(mpicxx libmpi.a libmpi.so /usr/lib64/mpi/gcc/openmpi2/bin /usr/lib64/mpi/gcc/openmpi2 /usr/lib64/mpi/gcc/openmpi2/include)
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
    NAME rccl-tests
    DESCRIPTION "Tests for the ROCm Communication Collectives Library"
    MAINTAINER "RCCL Maintainer <rccl-maintainer@amd.com>"
)
