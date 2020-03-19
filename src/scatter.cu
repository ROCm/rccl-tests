/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <hip/hip_runtime.h>
#include "common.h"

//#define DEBUG_PRINT

void print_header() {
  PRINT("# %10s  %12s  %6s  %6s            out-of-place                       in-place          \n", "", "", "", "");
  PRINT("# %10s  %12s  %6s  %6s  %7s  %6s  %6s  %5s  %7s  %6s  %6s  %5s\n", "size", "count", "type", "redop",
        "time", "algbw", "busbw", "error", "time", "algbw", "busbw", "error");
  PRINT("# %10s  %12s  %6s  %6s  %7s  %6s  %6s  %5s  %7s  %6s  %6s  %5s\n", "(B)", "(elements)", "", "",
        "(us)", "(GB/s)", "(GB/s)", "", "(us)", "(GB/s)", "(GB/s)", "");
}

void print_line_header (size_t size, size_t count, const char *typeName, const char *opName, int root) {
  PRINT("%12li  %12li  %6s  %6s", size, count, typeName, opName);
}

void ScatterGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, int nranks) {
  *sendcount = (count/nranks)*nranks;
  *recvcount = count/nranks;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
  *paramcount = *recvcount;
}

testResult_t ScatterInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  for (int i=0; i<args->nGpus; i++) {
    int gpuid = args->localRank*args->nThreads*args->nGpus + args->thread*args->nGpus + i;
    HIPCHECK(hipSetDevice(gpuid));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    HIPCHECK(hipMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    if (rank == root) {
      for (int j=0; j<nranks; j++) {
        TESTCHECK(InitData(((char*)data)+args->expectedBytes*j, recvcount, type, rep, j));
      }
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
    }
    TESTCHECK(InitData(args->expected[i], recvcount, type, rep, rank));
    HIPCHECK(hipDeviceSynchronize());
  }
  return testSuccess;
}

void ScatterGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize * (nranks - 1)) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = 1;
  *busBw = baseBw * factor;
}

testResult_t ScatterRunColl(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, hipStream_t stream) {
  int nRanks;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  size_t rankOffset = count * wordSize(type);
  if (count == 0) return testSuccess;

  int rank;
  NCCLCHECK(ncclCommUserRank(comm, &rank));
  NCCLCHECK(ncclGroupStart());
  if (rank == root) {
    for (int r=0; r<nRanks; r++)
      NCCLCHECK(ncclSend(((char*)sendbuff)+r*rankOffset, count, type, r, comm, stream));
  }
  NCCLCHECK(ncclRecv(recvbuff, count, type, root, comm, stream));
  NCCLCHECK(ncclGroupEnd());
  return testSuccess;
}

struct testColl scatterTest = {
  "Scatter",
  ScatterGetCollByteCount,
  ScatterInitData,
  ScatterGetBw,
  ScatterRunColl
};

void ScatterGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  ScatterGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, nranks);
}

testResult_t ScatterRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &scatterTest;
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
  ScatterGetBuffSize,
  ScatterRunTest
};
