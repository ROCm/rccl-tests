/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef __COMMON_H__
#define __COMMON_H__

#include "rccl/rccl.h"
#include <stdio.h>
#include <cstdint>
#include <algorithm>
#ifdef MPI_SUPPORT
#include "mpi.h"
#endif
#include <pthread.h>
#include "nccl1_compat.h"
#include "timer.h"

// For nccl.h < 2.13 since we define a weak fallback
extern "C" char const* ncclGetLastError(ncclComm_t comm);

#define HIPCHECK(cmd) do {                          \
  hipError_t e = cmd;                               \
  if( e != hipSuccess ) {                           \
    char hostname[1024];                            \
    getHostName(hostname, 1024);                    \
    printf("%s: Test HIP failure %s:%d '%s'\n",     \
         hostname,                                  \
        __FILE__,__LINE__,hipGetErrorString(e));    \
    return testCudaError;                           \
  }                                                 \
} while(0)

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,13,0)
#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    char hostname[1024];                            \
    getHostName(hostname, 1024);                    \
    printf("%s: Test NCCL failure %s:%d "           \
           "'%s / %s'\n",                           \
           hostname,__FILE__,__LINE__,              \
           ncclGetErrorString(res),                 \
           ncclGetLastError(NULL));                 \
    return testNcclError;                           \
  }                                                 \
} while(0)
#else
#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    char hostname[1024];                            \
    getHostName(hostname, 1024);                    \
    printf("%s: Test NCCL failure %s:%d '%s'\n",    \
         hostname,                                  \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    return testNcclError;                           \
  }                                                 \
} while(0)
#endif

typedef enum {
  testSuccess = 0,
  testInternalError = 1,
  testCudaError = 2,
  testNcclError = 3,
  testTimeout = 4,
  testNumResults = 5
} testResult_t;

// Relay errors up and trace
#define TESTCHECK(cmd) do {                         \
  testResult_t r = cmd;                             \
  if (r!= testSuccess) {                            \
    char hostname[1024];                            \
    getHostName(hostname, 1024);                    \
    printf(" .. %s pid %d: Test failure %s:%d\n",   \
         hostname, getpid(),                        \
        __FILE__,__LINE__);                         \
    return r;                                       \
  }                                                 \
} while(0)

struct testColl {
  const char name[20];
  void (*getCollByteCount)(
      size_t *sendcount, size_t *recvcount, size_t *paramcount,
      size_t *sendInplaceOffset, size_t *recvInplaceOffset,
      size_t count, int nranks);
  testResult_t (*initData)(struct threadArgs* args, ncclDataType_t type,
      ncclRedOp_t op, int root, int rep, int in_place);
  void (*getBw)(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks);
  testResult_t (*runColl)(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type,
      ncclRedOp_t op, int root, ncclComm_t comm, hipStream_t stream);
};
extern struct testColl allReduceTest;
extern struct testColl allGatherTest;
extern struct testColl reduceScatterTest;
extern struct testColl broadcastTest;
extern struct testColl reduceTest;
extern struct testColl alltoAllTest;

struct testEngine {
  void (*getBuffSize)(size_t *sendcount, size_t *recvcount, size_t count, int nranks);
  testResult_t (*runTest)(struct threadArgs* args, int root, ncclDataType_t type,
      const char* typeName, ncclRedOp_t op, const char* opName);
};

extern struct testEngine ncclTestEngine;

struct threadArgs {
  size_t nbytes;
  size_t minbytes;
  size_t maxbytes;
  size_t stepbytes;
  size_t stepfactor;

  int totalProcs;
  int nProcs;
  int proc;
  int nThreads;
  int thread;
  int nGpus;
  int* gpus;
  int localRank;
  int localNumDevices;
  int enable_multiranks;
  int nRanks;
  void** sendbuffs;
  size_t sendBytes;
  size_t sendInplaceOffset;
  void** recvbuffs;
  size_t recvInplaceOffset;
  ncclUniqueId ncclId;
  ncclComm_t* comms;
  hipStream_t* streams;

  void** expected;
  size_t expectedBytes;
  int* errors;
  double* bw;
  int* bw_count;

  int reportErrors;

  struct testColl* collTest;
};

typedef testResult_t (*threadFunc_t)(struct threadArgs* args);
struct testThread {
  pthread_t thread;
  threadFunc_t func;
  struct threadArgs args;
  testResult_t ret;
};

// Provided by common.cu
extern void Barrier(struct threadArgs* args);
extern testResult_t TimeTest(struct threadArgs* args, ncclDataType_t type, const char* typeName, ncclRedOp_t op,  const char* opName, int root);
extern testResult_t InitDataReduce(void* data, const size_t count, const size_t offset, ncclDataType_t type, ncclRedOp_t op, const uint64_t seed, const int nranks);
extern testResult_t InitData(void* data, const size_t count, size_t offset, ncclDataType_t type, ncclRedOp_t op, const uint64_t seed, const int nranks, const int rank);
extern void AllocateBuffs(void **sendbuff, void **recvbuff, void **expected, void **expectedHost, size_t nbytes, int nranks);

#include <unistd.h>

static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
      hostname[i] = '\0';
      return;
    }
  }
}

#include <stdint.h>

static uint64_t getHostHash(const char* string) {
  // Based on DJB2, result = result * 33 + char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) + string[c];
  }
  return result;
}

static size_t wordSize(ncclDataType_t type) {
  switch(type) {
    case ncclChar:
#if NCCL_MAJOR >= 2
    //case ncclInt8:
    case ncclUint8:
#endif
      return 1;
    case ncclHalf:
#if NCCL_MAJOR >= 2 && RCCL_BFLOAT16 == 1
    case ncclBfloat16:
#endif
    //case ncclFloat16:
      return 2;
    case ncclInt:
    case ncclFloat:
#if NCCL_MAJOR >= 2
    //case ncclInt32:
    case ncclUint32:
    //case ncclFloat32:
#endif
      return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclDouble:
    //case ncclFloat64:
      return 8;
    default: return 0;
  }
}

extern int test_ncclVersion; // init'd with ncclGetVersion()
typedef enum { ncclCoarse        = 0,
               ncclFine          = 1,
               ncclHost          = 2,
               ncclManaged       = 3,
               nccl_NUM_MTYPES   = 4 } ncclMemoryType_t;
extern const char *test_memorytypes[nccl_NUM_MTYPES];
constexpr int test_opNumMax = (int)ncclNumOps + (NCCL_VERSION_CODE >= NCCL_VERSION(2,11,0) ? 1 : 0);
extern int test_opnum;
extern int test_typenum;
extern ncclDataType_t test_types[ncclNumTypes];
extern const char *test_typenames[ncclNumTypes];
extern ncclRedOp_t test_ops[];
extern const char *test_opnames[];

static int ncclstringtotype(char *str) {
    for (int t=0; t<ncclNumTypes; t++) {
      if (strcmp(str, test_typenames[t]) == 0) {
        return t;
      }
    }
    if (strcmp(str, "all") == 0) {
      return -1;
    }
    printf("invalid type %s, defaulting to %s .. \n", str, test_typenames[ncclFloat]);
    return ncclFloat;
}

static int ncclstringtoop (char *str) {
    for (int o=0; o<test_opnum; o++) {
      if (strcmp(str, test_opnames[o]) == 0) {
        return o;
      }
    }
    if (strcmp(str, "all") == 0) {
      return -1;
    }
    printf("invalid op %s, defaulting to %s .. \n", str, test_opnames[ncclSum]);
    return ncclSum;
}

static int ncclstringtomtype (char *str) {
    for (int o=0; o<nccl_NUM_MTYPES; o++) {
      if (strcmp(str, test_memorytypes[o]) == 0) {
        return o;
      }
    }
    printf("invalid memorytype %s, defaulting to %s .. \n", str, test_memorytypes[ncclCoarse]);
    return ncclCoarse;
}

extern int is_main_proc;
extern thread_local int is_main_thread;
#define PRINT if (is_main_thread) printf

#endif
