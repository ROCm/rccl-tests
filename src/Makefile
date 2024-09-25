#
# Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
# Modifications are Copyright (c) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
#

ROCM_PATH ?= /opt/rocm
MPI_HOME ?= /usr/lib/openmpi
PREFIX ?= /usr/local
VERBOSE ?= 0
DEBUG ?= 0
NCCL_HOME ?= ""

HIPCC = $(ROCM_PATH)/bin/hipcc
CXX = $(HIPCC)

HIPCUFLAGS := -std=c++14
LDFLAGS    :=
HIPLDFLAGS :=

ifneq ($(NCCL_HOME), "")
HIPCUFLAGS += -I$(NCCL_HOME)/ -I$(NCCL_HOME)/include
HIPLDFLAGS   += -Wl,-rpath,$(NCCL_HOME) -L$(NCCL_HOME) -L$(NCCL_HOME)/lib
endif
HIPCUFLAGS += -I$(ROCM_PATH)/include
HIPCUFLAGS += -I$(ROCM_PATH)/include/hip
LDFLAGS    += -L$(ROCM_PATH)/lib -lhsa-runtime64 -lrt
HIPLDFLAGS += -L$(CUSTOM_RCCL_LIB) -L$(ROCM_PATH)/lib -lhsa-runtime64 -lrt -pthread

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
HIPIFY_DIR ?= $(BUILDDIR)/hipify

.PRECIOUS: $(HIPIFY_DIR)/%.cu.cpp $(HIPIFY_DIR)/%.h

ifeq ($(MPI), 1)
HIPCUFLAGS += -DMPI_SUPPORT -I${MPI_HOME}/include -I${MPI_HOME}/include/mpi
HIPLDFLAGS += -L${MPI_HOME}/lib -lmpi
else ifeq ($(MPICH), 1)
HIPCUFLAGS += -DMPI_SUPPORT -I/usr/include/mpich -I/usr/include/x86_64-linux-gnu/mpich -I/usr/include/mpich-x86_64/
HIPLDFLAGS += -L/usr/lib -L/usr/lib64/mpich/lib/ -lmpich
endif

LIBRARIES += rccl
HIPLDFLAGS   += $(LIBRARIES:%=-l%)

DST_DIR := $(BUILDDIR)
SRC_FILES := $(wildcard *.cu)
OBJ_FILES := $(SRC_FILES:%.cu=${DST_DIR}/%.o)
BIN_FILES_LIST := all_reduce all_gather broadcast reduce_scatter reduce alltoall scatter gather sendrecv alltoallv
BIN_FILES := $(BIN_FILES_LIST:%=${DST_DIR}/%_perf)

GIT_VERSION_FILE := ${DST_DIR}/src/git_version.cpp
GIT_REV          := $(shell git log --pretty=format:'%h' -n 1)
GIT_DIFF         := $(shell git diff --quiet --exit-code || echo +)
GIT_BRANCH       := $(shell git rev-parse --abbrev-ref HEAD)

build: ${BIN_FILES}

clean:
	rm -rf ${DST_DIR}

TEST_VERIFIABLE_SRCDIR := ../verifiable
TEST_VERIFIABLE_BUILDDIR := $(BUILDDIR)/verifiable
include ../verifiable/verifiable.mk

# Rule to create git_version.cpp
$(GIT_VERSION_FILE):
	@mkdir -p ${DST_DIR}/src
	@echo 'const char* rcclTestsGitHash = "$(GIT_BRANCH):$(GIT_REV)$(GIT_DIFF)";' > $@

${HIPIFY_DIR}/%.cu.cpp: %.cu
	@printf "Hipifying  %-35s > %s\n" $< $@
	@mkdir -p ${HIPIFY_DIR}
	hipify-perl -quiet-warnings $< > $@

${HIPIFY_DIR}/%.h: %.h
	@printf "Hipifying  %-35s > %s\n" $< $@
	@mkdir -p ${HIPIFY_DIR}
	hipify-perl -quiet-warnings $< > $@

${DST_DIR}/%.o: ${HIPIFY_DIR}/%.cu.cpp ${HIPIFY_DIR}/common.h $(TEST_VERIFIABLE_HDRS) $(GIT_VERSION_FILE)
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	echo "$(HIPCC) -o $@ $(HIPCUFLAGS) -I. -c $<"
	$(HIPCC) -o $@ $(HIPCUFLAGS) -I. -c $<

${DST_DIR}/timer.o: timer.cc timer.h
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(CXX) $(CXXFLAGS) -o $@ -c timer.cc

${DST_DIR}/%_perf:${DST_DIR}/%.o ${DST_DIR}/common.o ${DST_DIR}/timer.o $(TEST_VERIFIABLE_OBJS) $(DST_DIR)/src/git_version.cpp
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	echo "$(HIPCC) -o $@ $(HIPCUFLAGS) $^ ${HIPLDFLAGS}"
	$(HIPCC) -o $@ $(HIPCUFLAGS) $^ ${HIPLDFLAGS}

