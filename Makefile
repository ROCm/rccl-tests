#
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#
# See LICENCE.txt for license information
#

BUILDDIR ?= build
override BUILDDIR := $(abspath $(BUILDDIR))

.PHONY: all clean

default: src.build

TARGETS=$(filter-out src/hypercube.cu src/alltoallv.cu, $(wildcard src/*))

all:   ${TARGETS:%=%.build}
clean: ${TARGETS:%=%.clean}

%.build:
	${MAKE} -C $* build BUILDDIR=${BUILDDIR}

%.clean:
	${MAKE} -C $* clean BUILDDIR=${BUILDDIR}
