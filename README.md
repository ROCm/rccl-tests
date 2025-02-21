# RCCL Tests

These tests check both the performance and the correctness of RCCL operations. They can be compiled against [RCCL](https://github.com/ROCmSoftwarePlatform/rccl).

## Build

To build the tests, just type `make`.

If HIP is not installed in /opt/rocm, you may specify HIP\_HOME. Similarly, if RCCL is not installed in /usr, you may specify NCCL\_HOME and CUSTOM\_RCCL\_LIB.

```shell
$ make HIP_HOME=/path/to/hip NCCL_HOME=/path/to/rccl CUSTOM_RCCL_LIB=/path/to/rccl/lib/librccl.so
```

RCCL tests rely on MPI to work on multiple processes, hence multiple nodes. If you want to compile the tests with MPI support, you need to set MPI=1 and set MPI\_HOME to the path where MPI is installed.

```shell
$ make MPI=1 MPI_HOME=/path/to/mpi HIP_HOME=/path/to/hip NCCL_HOME=/path/to/rccl
```

RCCL tests can also be built using cmake. A typical sequence will be:

```shell
$ mkdir build
$ cd build
$ CXX=/opt/rocm/bin/hipcc cmake -DCMAKE_PREFIX_PATH=/path/to/rccl ..
$ make
```

When using the cmake build procedure, please make sure that RCCL has also been built using cmake (i.e. not using the install.sh script), since cmake will check
for cmake target and config files that are created during the RCCL build.

Using the cmake method also has the advantage that the build is automatically checking for MPI installations. The tests can be compiled with MPI support by adding the `-DUSE_MPI=ON` flag to the cmake command line. A user can request to use a particular MPI library by setting the environment variable `MPI_HOME` or add the path of the MPI library to the cmake prefix path with `-DCMAKE_PREFIX_PATH`.


## Usage

RCCL tests can run on multiple processes, multiple threads, and multiple HIP devices per thread. The number of process is managed by MPI and is therefore not passed to the tests as argument. The total number of ranks (=HIP devices) will be equal to (number of processes)\*(number of threads)\*(number of GPUs per thread).

### Quick examples

Run on 8 GPUs (`-g 8`), scanning from 8 Bytes to 128MBytes :
```shell
$ ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8
```

Run with MPI on 10 processes (potentially on multiple nodes) with 4 GPUs each, for a total of 40 GPUs:
```shell
$ mpirun -np 10 ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 4
```

For performance-oriented runs, on both single-node and multi-node, we suggest using 1 MPI process per GPU and `-g 1`. So, a run on 8 GPUs looks like :
```shell
$ mpirun -np 8 --bind-to numa ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
```
Running with 1 MPI process per GPU ensures a 1:1 mapping for CPUs and GPUs, which can be beneficial for smaller message sizes and better represents the real-world use of RCCL in Deep Learning frameworks like Pytorch and TensorFlow.

### Performance

See the [Performance](doc/PERFORMANCE.md) page for explanation about numbers, and in particular the "busbw" column.

### Arguments

All tests support the same set of arguments :

* Number of GPUs
  * `-t,--nthreads <num threads>` number of threads per process. Default : 1.
  * `-g,--ngpus <GPUs per thread>` number of gpus per thread. Default : 1.
* Sizes to scan
  * `-b,--minbytes <min size in bytes>` minimum size to start with. Default : 32M.
  * `-e,--maxbytes <max size in bytes>` maximum size to end at. Default : 32M.
  * Increments can be either fixed or a multiplication factor. Only one of those should be used
    * `-i,--stepbytes <increment size>` fixed increment between sizes. Default : 1M.
    * `-f,--stepfactor <increment factor>` multiplication factor between sizes. Default : disabled.
* RCCL operations arguments
  * `-o,--op <sum/prod/min/max/avg/all>` Specify which reduction operation to perform. Only relevant for reduction operations like Allreduce, Reduce or ReduceScatter. Default : Sum.
  * `-d,--datatype <nccltype/all>` Specify which datatype to use. Default : Float.
  * `-r,--root <root/all>` Specify which root to use. Only for operations with a root like broadcast or reduce. Default : 0.
  * `-y,--memory_type <coarse/fine/host/managed>` Default: Coarse
  * `-s,--stress_cycles <number of cycles>` Default: 1
  * `-u,--cumask <d0,d1,d2,d3>` Default: None
* Performance
  * `-n,--iters <iteration count>` number of iterations. Default : 20.
  * `-w,--warmup_iters <warmup iteration count>` number of warmup iterations (not timed). Default : 5.
  * `-m,--agg_iters <aggregation count>` number of operations to aggregate together in each iteration. Default : 1.
  * `-a,--average <0/1/2/3>` Report performance as an average across all ranks (MPI=1 only). <0=Rank0,1=Avg,2=Min,3=Max>. Default : 1.
* Test operation
  * `-p,--parallel_init <0/1>` use threads to initialize NCCL in parallel. Default : 0.
  * `-c,--check <check iteration count>` perform count iterations, checking correctness of results on each iteration. This can be quite slow on large numbers of GPUs. Default : 1.
  * `-z,--blocking <0/1>` Make NCCL collective blocking, i.e. have CPUs wait and sync after each collective. Default : 0.
  * `-G,--cudagraph <num graph launches>` Capture iterations as a CUDA graph and then replay specified number of times. Default : 0.
  * `-F,--cache_flush <cache flush after every -F iteration>` Enable cache flush after every -F iteration. Default : 0 (No cache flush).

## Unit tests

Unit tests for rccl-tests are implemented with pytest (python3 is also required).  Several notes for the unit tests:

1. The LD_LIBRARY_PATH environment variable will need to be set to include /path/to/rccl-install/lib/ in order to run the unit tests.
2. The HSA_FORCE_FINE_GRAIN_PCIE environment variable will need to be set to 1 in order to run the unit tests which use fine-grained memory type.

The unit tests can be invoked within the rccl-tests root, or in the test subfolder.  An example call to the unit tests:
```shell
$ LD_LIBRARY_PATH=/path/to/rccl-install/lib/ HSA_FORCE_FINE_GRAIN_PCIE=1 python3 -m pytest
```

## Copyright

RCCL tests are provided under the BSD license.

All source code and accompanying documentation is copyright (c) 2016-2021, NVIDIA CORPORATION. All rights reserved.

All modifications are copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.

