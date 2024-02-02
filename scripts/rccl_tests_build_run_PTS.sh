#!/bin/bash
echo "This script is for building and running the rccl-tests as well as Unit tests"
echo "Please ensure that the following environment variables are pointing to correct directions!"

########## Set the appropriate directories ##########
export _HIP_HOME=/opt/rocm/hip
export _MPI_HOME=/path/to/mpi/build
export _RCCL_HOME=/opt/rocm/rccl/build

export LD_LIBRARY_PATH=$_MPI_HOME/lib:$LD_LIBRARY_PATH
export PATH=$_MPI_HOME/bin/:$PATH
echo "HIP_HOME=$_HIP_HOME"
echo "MPI_HOME=$_MPI_HOME"
echo "RCCL_HOME=$_RCCL_HOME"

echo "########## Print the system information ##########"
sudo dmidecode | grep "Product Name"
rocm-smi --showtopo

########## Set the number of GPUs ##########
ngpus=8
set -x
########## Build the RCCL-tests benchmark ##########
echo "Do you want to run tests on multiple nodes?"
read -p '(y/n) ' RESPONSE
if [ "$RESPONSE" = "y" ]; then

        ########## MPI Installation check ##########
        MPI_Installed=$(which mpicc)
        
        if [ -z "$MPI_Installed" ]; then
                echo "MPI is not installed! Install MPI and set the PATH environment variable to include PATH=/path/to/MPI-install/bin/:$PATH";
                exit
        else
                cd ..
                rm -rf rccl-tests
                git clone https://github.com/ROCmSoftwarePlatform/rccl-tests.git
                cd rccl-tests
                make MPI=1 MPI_HOME=$_MPI_HOME HIP_HOME=$_HIP_HOME NCCL_HOME=$_RCCL_HOME
        fi
else
        cd ..
        rm -rf rccl-tests
        git clone https://github.com/ROCmSoftwarePlatform/rccl-tests.git
        cd rccl-tests
        make HIP_HOME=$_HIP_HOME NCCL_HOME=$_RCCL_HOME
fi       

########## Run the RCCL-tests benchmark ##########
cd build
echo "Allreduce Test"
./all_reduce_perf -b 8 -e 1G -f 2 -g $ngpus
echo "Broadcast Test"
./broadcast_perf -b 8 -e 1G -f 2 -g $ngpus
echo "Reduce Test"
./reduce_perf -b 8 -e 1G -f 2 -g $ngpus
echo "Reduce_scatter Test"
./reduce_scatter_perf -b 8 -e 1G -f 2 -g $ngpus
echo "Allgather Test"
./all_gather_perf -b 8 -e 1G -f 2 -g $ngpus
echo "Send_Recv Test"
./sendrecv_perf -b 8 -e 1G -f 2 -g $ngpus
echo "Scatter Test"
./scatter_perf -b 8 -e 1G -f 2 -g $ngpus
echo "Gather Test"
./gather_perf -b 8 -e 1G -f 2 -g $ngpus
echo "Alltoall Test"
./alltoall_perf -b 8 -e 1G -f 2 -g $ngpus
echo "Alltoallv Test"
./alltoallv_perf -b 8 -e 1G -f 2 -g $ngpus



