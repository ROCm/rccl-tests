/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <hip/hip_runtime.h>
#include "common.h"

//#define DEBUG_PRINT
#define USE_RCCL_GATHER_SCATTER

void print_header() {
  PRINT("# %10s  %12s  %6s            out-of-place                       in-place          \n", "", "", "");
  PRINT("# %10s  %12s  %6s  %7s  %6s  %6s  %5s  %7s  %6s  %6s  %5s\n", "size", "count", "type",
        "time", "algbw", "busbw", "error", "time", "algbw", "busbw", "error");
  PRINT("# %10s  %12s  %6s  %7s  %6s  %6s  %5s  %7s  %6s  %6s  %5s\n", "(B)", "(elements)", "",
        "(us)", "(GB/s)", "(GB/s)", "", "(us)", "(GB/s)", "(GB/s)", "");
}

void print_line_header (size_t size, size_t count, const char *typeName, const char *opName, int root) {
  PRINT("%12li  %12li  %6s", size, count, typeName);
}

void GatherGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, int nranks) {
  *sendcount = count/nranks;
  *recvcount = (count/nranks)*nranks;
  *sendInplaceOffset = count/nranks;
  *recvInplaceOffset = 0;
  *paramcount = *sendcount;
}

testResult_t GatherInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  for (int i=0; i<args->nGpus; i++) {
    int gpuid = args->localRank*args->nThreads*args->nGpus + args->thread*args->nGpus + i;
    HIPCHECK(hipSetDevice(gpuid));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    HIPCHECK(hipMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? ((char*)args->recvbuffs[i])+rank*args->sendBytes : args->sendbuffs[i];
    TESTCHECK(InitData(data, sendcount, type, rep, rank));
#ifdef DEBUG_PRINT
    int *dataHost = (int *)malloc(args->sendBytes);
    hipMemcpy(dataHost, data, args->sendBytes, hipMemcpyDeviceToHost);
    printf("\n Rank [%d] Init: ", rank);
    for (int j=0; j<args->sendBytes/sizeof(int); j++) {
     printf("%d:%d ", j, dataHost[j]);
    }
    printf("\n");
    free(dataHost);
#endif
    for (int j=0; j<nranks; j++) {
      TESTCHECK(InitData(((char*)args->expected[i])+args->sendBytes*j, sendcount, type, rep, j));
    }
    HIPCHECK(hipDeviceSynchronize());
  }
  // We don't support in-place gather
  args->reportErrors = in_place ? 0 : 1;
  return testSuccess;
}

void GatherGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize * (nranks - 1)) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = 1;
  *busBw = baseBw * factor;
}

testResult_t GatherRunColl(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, hipStream_t stream) {
  int nRanks;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  size_t rankOffset = count * wordSize(type);
  if (count == 0) return testSuccess;

  int rank;
  NCCLCHECK(ncclCommUserRank(comm, &rank));
#if NCCL_MAJOR >= 2 && NCCL_MINOR >= 7
#if defined(RCCL_GATHER_SCATTER) && defined(USE_RCCL_GATHER_SCATTER)
  if (rank == root)
    NCCLCHECK(ncclGather(sendbuff, recvbuff, count, type, root, comm, stream));
  else
    NCCLCHECK(ncclGather(sendbuff, 0, count, type, root, comm, stream));
#else
  NCCLCHECK(ncclGroupStart());
  if (rank == root) {
  for (int r=0; r<nRanks; r++)
    NCCLCHECK(ncclRecv(((char*)recvbuff)+r*rankOffset, count, type, r, comm, stream));
  }
  NCCLCHECK(ncclSend(sendbuff, count, type, root, comm, stream));
  NCCLCHECK(ncclGroupEnd());
#endif
#endif
  return testSuccess;
}

struct testColl gatherTest = {
  "Gather",
  GatherGetCollByteCount,
  GatherInitData,
  GatherGetBw,
  GatherRunColl
};

void GatherGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  GatherGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, nranks);
}

testResult_t GatherRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &gatherTest;
  ncclDataType_t *run_types;
  const char **run_typenames;
  int type_count;

  if ((int)type != -1) {
    type_count = 1;
    run_types = &type;
    run_typenames = &typeName;
  } else {
    type_count = ncclNumTypes;
    run_types = test_types;
    run_typenames = test_typenames;
  }

  for (int i=0; i<type_count; i++) {
    TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "", root));
  }
  return testSuccess;
}

struct testEngine ncclTestEngine = {
  GatherGetBuffSize,
  GatherRunTest
};
