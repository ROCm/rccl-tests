# Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
# Modifications Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information


# We requires both of the following paths to be set upon including this makefile
# TEST_VERIFIABLE_SRCDIR = <points to this directory>
# TEST_VERIFIABLE_BUILDDIR = <points to destination of .o file>

TEST_VERIFIABLE_HDRS = $(TEST_VERIFIABLE_SRCDIR)/verifiable.h
TEST_VERIFIABLE_OBJS = $(TEST_VERIFIABLE_BUILDDIR)/verifiable.o

$(TEST_VERIFIABLE_BUILDDIR)/verifiable.o: $(TEST_VERIFIABLE_SRCDIR)/verifiable.cu $(TEST_VERIFY_REDUCE_HDRS)
	@printf "Compiling %s\n" $@
	@mkdir -p $(TEST_VERIFIABLE_BUILDDIR)
	echo " $(HIPCC) -o $@ $(HIPCUFLAGS) -c $(TEST_VERIFIABLE_SRCDIR)/verifiable.cu"
	$(HIPCC) -o $@ $(HIPCUFLAGS) -c $(TEST_VERIFIABLE_SRCDIR)/verifiable.cu
