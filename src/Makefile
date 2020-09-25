#
# Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
# Modifications are Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
#

ROCM_HOME ?= /opt/rocm
MPI_HOME ?= /usr/lib/openmpi
PREFIX ?= /usr/local
VERBOSE ?= 0
DEBUG ?= 0
NCCL_HOME ?= ""

HIPCC = $(ROCM_HOME)/hip/bin/hipcc
CXX = $(HIPCC)

HIPCUFLAGS := -std=c++14
LDFLAGS    :=
HIPLDFLAGS :=

ifneq ($(NCCL_HOME), "")
HIPCUFLAGS += -I$(NCCL_HOME)
HIPLDFLAGS   += -Wl,-rpath,$(NCCL_HOME) -L$(NCCL_HOME)
endif
HIPCUFLAGS += -I$(ROCM_HOME)/include
HIPCUFLAGS += -I$(ROCM_HOME)/include/rccl
HIPCUFLAGS += -I$(ROCM_HOME)/hip/include/hip
LDFLAGS    += -L$(ROCM_HOME)/lib -lhsa-runtime64 -lrt
HIPLDFLAGS += $(CUSTOM_RCCL_LIB) -L$(ROCM_HOME)/lib -lhsa-runtime64 -lrt

ifeq ($(DEBUG), 0)
HIPCUFLAGS += -O3
else
HIPCUFLAGS += -O0 -g -ggdb3
endif

ifeq ($(VERBOSE), 0)
.SILENT:
endif

.PHONY: build clean

BUILDDIR ?= ../build

ifeq ($(MPI), 1)
HIPCUFLAGS += -DMPI_SUPPORT -I${MPI_HOME}/include -I${MPI_HOME}/include/mpi
HIPLDFLAGS += -L${MPI_HOME}/lib -lmpi
else ifeq ($(MPICH), 1)
HIPCUFLAGS += -DMPI_SUPPORT -I/usr/include/mpich
HIPLDFLAGS += -L/usr/lib -lmpi
endif

LIBRARIES += rccl
HIPLDFLAGS   += $(LIBRARIES:%=-l%)

DST_DIR := $(BUILDDIR)
SRC_FILES := $(wildcard *.cu)
OBJ_FILES := $(SRC_FILES:%.cu=${DST_DIR}/%.o)
BIN_FILES_LIST := all_reduce all_gather broadcast reduce_scatter reduce alltoall gather scatter sendrecv alltoallv
BIN_FILES := $(BIN_FILES_LIST:%=${DST_DIR}/%_perf)

build: ${BIN_FILES}

clean:
	rm -rf ${DST_DIR}

${DST_DIR}/%.o: %.cu common.h
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	echo "$(HIPCC) -o $@ $(HIPCUFLAGS) -c $<"
	$(HIPCC) -o $@ $(HIPCUFLAGS) -c $<

${DST_DIR}/%_perf:${DST_DIR}/%.o ${DST_DIR}/common.o
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	echo "$(HIPCC) -o $@ $(HIPCUFLAGS) $^ ${HIPLDFLAGS}"
	$(HIPCC) -o $@ $(HIPCUFLAGS) $^ ${HIPLDFLAGS}

