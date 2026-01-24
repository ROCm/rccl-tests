#!/bin/bash
# RCCL Benchmark Analysis Runner
#
# Usage: ./run_analysis.sh [benchmark] [options]
#
# Example: ./run_analysis.sh all_reduce -b 8 -e 1g -f 2
#
# This script:
# 1. Runs the specified benchmark under rocprofv3 with parquet output
# 2. Runs analysis scripts to generate plots and statistics
# 3. Opens the results (if display available)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCPROFV3="/work/lmeadows/rocm-systems/rocprofiler-sdk-build/bin/rocprofv3"
MPI_DIR="/opt/openmpi-4.1.5"
OUTPUT_DIR="rocp"
ANALYSIS_DIR="analysis_output"

# Default benchmark
BENCHMARK="${1:-all_reduce}"
shift 2>/dev/null || true

# Default MPI ranks
NP="${NP:-4}"

# Benchmark executable
BENCHMARK_EXE="build/${BENCHMARK}_perf"

if [[ ! -f "$BENCHMARK_EXE" ]]; then
    echo "Error: Benchmark executable not found: $BENCHMARK_EXE"
    echo "Available benchmarks:"
    ls build/*_perf 2>/dev/null | xargs -n1 basename | sed 's/_perf$//'
    exit 1
fi

# Clean previous output
echo "Cleaning previous output..."
rm -rf "$OUTPUT_DIR" rank_*_timestamps.csv "$ANALYSIS_DIR"

# Set up environment
export LD_LIBRARY_PATH="$MPI_DIR/lib:$LD_LIBRARY_PATH"

# Construct the benchmark command
BENCH_ARGS="${@:--g 1 -b 8 -e 1g -f 2}"

echo "=============================================="
echo "RCCL Benchmark Analysis"
echo "=============================================="
echo "Benchmark: $BENCHMARK"
echo "Ranks: $NP"
echo "Arguments: $BENCH_ARGS"
echo "=============================================="
echo ""

# Run the benchmark
echo "Running benchmark with rocprofv3..."
"$MPI_DIR/bin/mpirun" -np "$NP" \
    "$ROCPROFV3" --kernel-trace -f parquet -d "$OUTPUT_DIR" -- \
    "$BENCHMARK_EXE" $BENCH_ARGS

echo ""
echo "Benchmark complete!"
echo ""

# Find the parquet directory (hostname subdirectory)
PARQUET_SUBDIR=$(ls -d "$OUTPUT_DIR"/*/ 2>/dev/null | head -1)
if [[ -z "$PARQUET_SUBDIR" ]]; then
    echo "Error: No parquet output found in $OUTPUT_DIR"
    exit 1
fi

echo "Parquet data in: $PARQUET_SUBDIR"
echo ""

# Run analysis
echo "Running main analysis..."
python3 "$SCRIPT_DIR/analyze_rccl.py" \
    -p "$PARQUET_SUBDIR" \
    -o "$ANALYSIS_DIR" \
    --rank-comparison-sizes 65536 1048576 67108864

echo ""
echo "Running kernel analysis..."
python3 "$SCRIPT_DIR/analyze_kernels.py" \
    -p "$PARQUET_SUBDIR" \
    -o "$ANALYSIS_DIR"

echo ""
echo "=============================================="
echo "Analysis Complete!"
echo "=============================================="
echo ""
echo "Output files:"
ls -la "$ANALYSIS_DIR"
echo ""
echo "Key files:"
echo "  - $ANALYSIS_DIR/benchmark_statistics.csv"
echo "  - $ANALYSIS_DIR/kernel_statistics.csv"
echo "  - $ANALYSIS_DIR/${BENCHMARK}_scaling.png"
echo ""

