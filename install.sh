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
    echo "    [--rccl_home] Specify custom path for RCCL installation (default: /opt/rocm/rccl)"
    echo "    [--mpi_home] Specify path to your MPI installation."
}

# #################################################
# global variables
# #################################################
run_tests=false
build_release=true
mpi_enabled=false
rccl_dir=/opt/rocm/rccl
mpi_dir=""
# #################################################
# Parameter parsing
# #################################################

# check if we have a modern version of getopt that can handle whitespace and long parameters
getopt -T
if [[ $? -eq 4 ]]; then
    GETOPT_PARSE=$(getopt --name "${0}" --longoptions help,mpi,test,rccl_home:,mpi_home: --options hmt -- "$@")
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
	--) shift ; break ;;
	*)  echo "Unexpected command line parameter received; aborting";
	    exit 1
	    ;;
    esac
    done

# Install the pre-commit hook
#bash ./githooks/install

build_dir=./build
# #################################################
# prep
# #################################################
# ensure a clean build environment
rm -rf ${build_dir}

if ($mpi_enabled); then
    if [[ ${mpi_dir} -eq "" ]]; then
        echo "MPI flag enabled but path to MPI installation not specified.  See --mpi_home command line argument."
        exit 1
    else
        make NCCL_HOME=${rccl_dir} CUSTOM_RCCL_LIB=${rccl_dir}/lib/librccl.so MPI=1 MPI_HOME=${mpi_dir} -j$(nproc)
    fi
else
    make NCCL_HOME=${rccl_dir} CUSTOM_RCCL_LIB=${rccl_dir}/lib/librccl.so -j$(nproc)
fi

# Optionally, run tests if they're enabled.
if ($run_tests); then
    if ($mpi_enabled); then
        cd test; LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${rccl_dir}/lib:${mpi_dir}/lib PATH=$PATH:${mpi_dir}/bin python3 -m pytest
    else
        cd test; LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${rccl_dir}/lib python3 -m pytest
    fi
fi
