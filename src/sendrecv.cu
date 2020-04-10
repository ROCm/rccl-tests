/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <hip/hip_runtime.h>
#include "common.h"

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

void SendRecvGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, int nranks) {
  *sendcount = count;
  *recvcount = count;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
  *paramcount = *sendcount;
}

testResult_t SendRecvInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  for (int i=0; i<args->nGpus; i++) {
    int gpuid = args->localRank*args->nThreads*args->nGpus + args->thread*args->nGpus + i;
    HIPCHECK(hipSetDevice(gpuid));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    TESTCHECK(InitData(args->sendbuffs[i], sendcount, type, rep, rank));
    TESTCHECK(InitData(args->recvbuffs[i], recvcount, type, rep, rank));
    int src = rank < nranks/2 ? rank : rank - nranks/2;
    TESTCHECK(InitData(args->expected[i],  recvcount, type, rep, src));
    HIPCHECK(hipDeviceSynchronize());
  }
  return testSuccess;
}

void SendRecvGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = nranks/2;
  *busBw = baseBw * factor;
}

testResult_t SendRecvRunColl(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, hipStream_t stream) {
  int rank, peer, nranks, npairs;
  NCCLCHECK(ncclCommUserRank(comm, &rank));
  NCCLCHECK(ncclCommCount(comm, &nranks));
  npairs = nranks / 2;
#if NCCL_MAJOR >= 2 && NCCL_MINOR >= 7
  if (rank < npairs) {
      peer = rank + npairs;
      NCCLCHECK(ncclSend(sendbuff, count, type, peer, comm, stream));
  } else if (rank < 2*npairs) {
      peer = rank - npairs;
      NCCLCHECK(ncclRecv(recvbuff, count, type, peer, comm, stream));
  }
#endif
  return testSuccess;
}

struct testColl sendrecvTest = {
  "SendRecv",
  SendRecvGetCollByteCount,
  SendRecvInitData,
  SendRecvGetBw,
  SendRecvRunColl
};

void SendRecvGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  SendRecvGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, nranks);
}

testResult_t SendRecvRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &sendrecvTest;
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
      TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "", 0));
  }
  return testSuccess;
}

struct testEngine ncclTestEngine = {
  SendRecvGetBuffSize,
  SendRecvRunTest
};
