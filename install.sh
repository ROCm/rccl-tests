#!/bin/bash
# Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.

# #################################################
# helper functions
# #################################################
function display_help()
{
    echo "RCCL-tests build & installation helper script"
    echo "./install [-h|--help] "
    echo "    [-h|--help] Prints this help message."
    echo "    [-m|--mpi] Build RCCL-tests with MPI support. (see --mpi_home below.)"
    echo "    [-t|--test] Run unit-tests after building RCCL-Tests."
    echo "    [--rccl_home] Specify custom path for RCCL installation (default: /opt/rocm)"
    echo "    [--mpi_home] Specify path to your MPI installation."
    echo "    [--hip_compiler] Specify path to HIP compiler (default: /opt/rocm/bin/amdclang++)"
    echo "    [--gpu_targets] Specify GPU targets (default:gfx906,gfx908,gfx90a,gfx942,gfx950,gfx1030,gfx1100,gxf1101,gfx1102,gfx1200,gfx1201)"
}

# #################################################
# global variables
# #################################################
run_tests=false
build_release=true
mpi_enabled=false
rccl_dir=/opt/rocm
mpi_dir=""
hip_compiler=/opt/rocm/bin/amdclang++
gpu_targets=""
# #################################################
# Parameter parsing
# #################################################

# check if we have a modern version of getopt that can handle whitespace and long parameters
getopt -T
if [[ $? -eq 4 ]]; then
    GETOPT_PARSE=$(getopt --name "${0}" --longoptions help,mpi,test,rccl_home:,mpi_home:,hip_compiler:,gpu_targets: --options hmt -- "$@")
else
    echo "Need a new version of getopt"
    exit 1
fi

if [[ $? -ne 0 ]]; then
    echo "getopt invocation failed; could not parse the command line";
    exit 1
fi

eval set -- "${GETOPT_PARSE}"

while true; do
    case "${1}" in
	-h|--help)
        display_help
        exit 0
        ;;
	-m|--mpi)
	    mpi_enabled=true
	    shift ;;
	-t|--test)
	    run_tests=true
	    shift ;;
    --rccl_home)
        rccl_dir=${2}
        shift 2 ;;
    --mpi_home)
        mpi_dir=${2}
        shift 2 ;;
    --hip_compiler)
        hip_compiler=${2}
        shift 2 ;;
    --gpu_targets)
        gpu_targets=${2}
        shift 2 ;;
	--) shift ; break ;;
	*)  echo "Unexpected command line parameter received; aborting";
	    exit 1
	    ;;
    esac
    done

# throw error code after running a command in the install script
check_exit_code( )
{
  if (( $1 != 0 )); then
    exit $1
  fi
}

# Install the pre-commit hook
#bash ./githooks/install

build_dir=./build
# #################################################
# prep
# #################################################
# ensure a clean build environment
rm -rf ${build_dir}

if ! command -v ${hip_compiler} 2>&1 >/dev/null ; then
    echo "HIP Compiler does not exist at ${hip_compiler}. Please check the path."
    echo "Defaulting to /opt/rocm/bin/amdclang++"
    hip_compiler=/opt/rocm/bin/amdclang++

    if ! command -v ${hip_compiler} 2>&1 >/dev/null ; then
        echo "${hip_compiler} does not exist. Please be advised."
	echo "Defaulting to /opt/rocm/bin/hipcc"
	hip_compiler=/opt/rocm/bin/hipcc

	if ! command -v ${hip_compiler} 2>&1 >/dev/null ; then
            echo "${hip_compiler} does not exist!. Please check your ROCm installation."
	    echo "Cannot proceed with building rccl-tests!"
	    exit 1
	fi
    fi
fi

if ($mpi_enabled); then
    if [[ ${mpi_dir} == "" ]]; then
        echo "MPI flag enabled but path to MPI installation not specified.  See --mpi_home command line argument."
        exit 1
    else
        make NCCL_HOME=${rccl_dir} CUSTOM_RCCL_LIB=${rccl_dir}/lib/librccl.so MPI=1 MPI_HOME=${mpi_dir} HIPCC=${hip_compiler} GPU_TARGETS=${gpu_targets} -j$(nproc)
    fi
else
    make NCCL_HOME=${rccl_dir} CUSTOM_RCCL_LIB=${rccl_dir}/lib/librccl.so HIP_COMPILER=${hip_compiler} GPU_TARGETS=${gpu_targets} -j$(nproc)
fi
check_exit_code "$?"

# Optionally, run tests if they're enabled.
if ($run_tests); then
    if ($mpi_enabled); then
        cd test; LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${rccl_dir}/lib:${mpi_dir}/lib PATH=$PATH:${mpi_dir}/bin python3 -m pytest
    else
        cd test; LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${rccl_dir}/lib python3 -m pytest -k "not MPI"
    fi
fi
