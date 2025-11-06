# Makefile for GPU Transfer Benchmark

ROCM_PATH ?= /opt/rocm
HIPCC ?= $(ROCM_PATH)/bin/amdclang++
HIPCONFIG = $(ROCM_PATH)/bin/hipconfig

# Get HIP version for GPU targets
HIP_VERSION = $(strip $(shell which $(HIPCONFIG) >/dev/null && $(HIPCONFIG) --version))
HIP_MAJOR = $(shell echo $(HIP_VERSION) | cut -d "." -f 1)
HIP_MINOR = $(shell echo $(HIP_VERSION) | cut -d "." -f 2)

# GPU targets (similar to RCCL tests)
ifndef GPU_TARGETS
GPU_TARGETS = gfx906 gfx908 gfx90a
  ifeq ($(shell test "0$(HIP_MAJOR)" -ge 7; echo $$?),0)
    GPU_TARGETS += gfx942 gfx950
  else ifeq ($(shell test "0$(HIP_MAJOR)" -eq 6; echo $$?),0)
    GPU_TARGETS += gfx942
    ifeq ($(shell test "0$(HIP_MINOR)" -ge 5; echo $$?),0)
    GPU_TARGETS += gfx950
    endif
  endif
GPU_TARGETS += gfx1030 gfx1100 gfx1101 gfx1102 gfx1200 gfx1201
endif

GPU_TARGETS_FLAGS = $(foreach target,$(GPU_TARGETS),"--offload-arch=$(target)")

# Compilation flags
HIPCUFLAGS := -std=c++14 -I$(ROCM_PATH)/include -I$(ROCM_PATH)/include/hip
HIPCUFLAGS += -x hip -D__HIP_PLATFORM_AMD__ -D__HIPCC__ $(GPU_TARGETS_FLAGS)
HIPCUFLAGS += -O3

# Linking flags
HIPLDFLAGS := -L$(ROCM_PATH)/lib -lhsa-runtime64 -lamdhip64 -lstdc++ -lrt -pthread

# Source and target
SRC = gpu_transfer_bench.cu
TARGET = gpu_transfer_bench

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCC) $(HIPCUFLAGS) $(HIPLDFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

help:
	@echo "GPU Transfer Benchmark Makefile"
	@echo "==============================="
	@echo "Targets:"
	@echo "  all     - Build the benchmark (default)"
	@echo "  clean   - Remove built files"
	@echo "  run     - Build and run the benchmark"
	@echo "  help    - Show this help"
	@echo ""
	@echo "Variables:"
	@echo "  ROCM_PATH  - ROCm installation path (default: /opt/rocm)"
	@echo "  GPU_TARGETS - GPU architectures to target"
	@echo ""
	@echo "Current settings:"
	@echo "  ROCM_PATH: $(ROCM_PATH)"
	@echo "  GPU_TARGETS: $(GPU_TARGETS)"
	@echo "  HIP_VERSION: $(HIP_VERSION)"
