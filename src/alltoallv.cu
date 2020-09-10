/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <hip/hip_runtime.h>
#include "common.h"

#define USE_RCCL_GATHER_SCATTER

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

void AlltoAllvGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, int nranks) {
  if (count < nranks*nranks/2) {
    *sendcount = 0;
    *recvcount = 0;
    *sendInplaceOffset = 0;
    *recvInplaceOffset = 0;
    *paramcount = 0;
  } else {
    *sendcount = (count/nranks)*nranks;
    *recvcount = (count/nranks)*nranks;
    *sendInplaceOffset = 0;
    *recvInplaceOffset = 0;
    *paramcount = count/nranks;
  }
}

testResult_t AlltoAllvInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  for (int i=0; i<args->nGpus; i++) {
    char* str = getenv("NCCL_TESTS_DEVICE");
    int gpuid = str ? atoi(str) : args->localRank*args->nThreads*args->nGpus + args->thread*args->nGpus + i;
    HIPCHECK(hipSetDevice(gpuid));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    HIPCHECK(hipMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    TESTCHECK(InitData(data, sendcount, type, rep, rank));
#if 0
    int *dataHost = (int *)malloc(args->sendBytes);
    hipMemcpy(dataHost, data, args->sendBytes, hipMemcpyDeviceToHost);
    printf(" Rank [%d] Original: ", rank);
    for(int j=0; j<sendcount; j++) {
      printf("%d:%d ", j, dataHost[j]);
    }
    printf("\n");
    free(dataHost);
#endif
    size_t rdisp = 0;
    size_t data_count = sendcount*2/nranks;
    size_t chunksize = data_count/nranks;
    for (int j=0; j<nranks; j++) {
      size_t scount = 0, rcount = ((j+rank)%nranks)*chunksize;
      if (j+rank == nranks-1)
          rcount += (sendcount-chunksize*(nranks-1)*nranks/2);
      size_t sdisp = 0;
      for (int k=0; k<nranks; k++) {
        scount = ((k+j)%nranks)*chunksize;
        if (k+j == nranks-1)
          scount += (sendcount-chunksize*(nranks-1)*nranks/2);
        if (k == rank)
          break;
        sdisp += scount;
      }
      TESTCHECK(InitData(((char*)args->expected[i])+rdisp*wordSize(type), rcount, type, rep+sdisp, j));
      rdisp += rcount;
    }
    HIPCHECK(hipDeviceSynchronize());
  }
  // We don't support in-place alltoall
  args->reportErrors = in_place ? 0 : 1;
  return testSuccess;
}

void AlltoAllvGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * nranks * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks-1))/((double)(nranks));
  *busBw = baseBw * factor;
}

testResult_t AlltoAllvRunColl(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, hipStream_t stream) {
  int nranks;
  NCCLCHECK(ncclCommCount(comm, &nranks));
  int rank;
  NCCLCHECK(ncclCommUserRank(comm, &rank));
  #define MAX_ALLTOALLV_RANKS 256
  static size_t sendcounts[MAX_ALLTOALLV_RANKS], recvcounts[MAX_ALLTOALLV_RANKS], sdispls[MAX_ALLTOALLV_RANKS], rdispls[MAX_ALLTOALLV_RANKS];
  if (count == 0) return testSuccess;
  if (nranks > MAX_ALLTOALLV_RANKS) {
    printf("Number of ranks %d exceeds limit %d\n", nranks, MAX_ALLTOALLV_RANKS);
    return testNcclError;
  }

  size_t disp = 0;
  size_t chunksize = count*2/nranks;
  for (int i = 0; i < nranks; i++) {
      size_t scount = ((i+rank)%nranks)*chunksize;
      if (i+rank == nranks-1)
          scount += (count*nranks-chunksize*(nranks-1)*nranks/2);
      sendcounts[i] = recvcounts[i] = scount;
      sdispls[i] = rdispls[i] = disp;
      disp += scount;
  }

#if NCCL_MAJOR < 2 || NCCL_MINOR < 7
  printf("NCCL 2.7 or later is needed for alltoallv. This test was compiled with %d.%d.\n", NCCL_MAJOR, NCCL_MINOR);
  return testNcclError;
#else
#if defined(RCCL_ALLTOALLV) && defined(USE_RCCL_GATHER_SCATTER)
  NCCLCHECK(ncclAllToAllv(sendbuff, sendcounts, sdispls, recvbuff, recvcounts, rdispls, type, comm, stream));
#else
  NCCLCHECK(ncclGroupStart());
  for (int r=0; r<nranks; r++) {
    if (sendcounts[r] != 0) {
      NCCLCHECK(ncclSend(
          ((char*)sendbuff) + sdispls[r] * wordSize(type),
          sendcounts[r],
          type,
          r,
          comm,
          stream));
    }
    if (recvcounts[r] != 0) {
      NCCLCHECK(ncclRecv(
          ((char*)recvbuff) + rdispls[r] * wordSize(type),
          recvcounts[r],
          type,
          r,
          comm,
          stream));
    }
  }
  NCCLCHECK(ncclGroupEnd());
#endif
  return testSuccess;
#endif
}

struct testColl alltoAllTest = {
  "AlltoAllv",
  AlltoAllvGetCollByteCount,
  AlltoAllvInitData,
  AlltoAllvGetBw,
  AlltoAllvRunColl
};

void AlltoAllvGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  AlltoAllvGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, nranks);
}

testResult_t AlltoAllvRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &alltoAllTest;
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
      TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "", -1));
  }
  return testSuccess;
}

struct testEngine ncclTestEngine = {
  AlltoAllvGetBuffSize,
  AlltoAllvRunTest
};
