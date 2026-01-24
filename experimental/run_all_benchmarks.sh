#!/bin/bash
# Run all RCCL benchmarks with rocprofv3 parquet output and analysis
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCPROFV3="/work/lmeadows/rocm-systems/rocprofiler-sdk-build/bin/rocprofv3"
MPI_DIR="/opt/openmpi-4.1.5"
RESULTS_BASE="/work/lmeadows/rccl/rccl-tests/benchmark_results"
NP=4

# Create timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${RESULTS_BASE}/${TIMESTAMP}"

mkdir -p "$RESULTS_DIR"

export LD_LIBRARY_PATH="$MPI_DIR/lib:$LD_LIBRARY_PATH"

# List of benchmarks
BENCHMARKS=(
    all_reduce
    all_gather
    broadcast
    reduce
    reduce_scatter
    alltoall
    alltoallv
    scatter
    gather
    sendrecv
    hypercube
    all_reduce_bias
)

# Common benchmark arguments
BENCH_ARGS="-g 1 -b 8 -e 128m -f 2"

echo "=============================================="
echo "Running ALL RCCL Benchmarks"
echo "=============================================="
echo "Timestamp: $TIMESTAMP"
echo "Results directory: $RESULTS_DIR"
echo "MPI ranks: $NP"
echo "Benchmark args: $BENCH_ARGS"
echo "=============================================="
echo ""

# Summary file
SUMMARY_FILE="$RESULTS_DIR/summary.txt"
echo "RCCL Benchmark Summary - $TIMESTAMP" > "$SUMMARY_FILE"
echo "=======================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for BENCH in "${BENCHMARKS[@]}"; do
    BENCH_EXE="build/${BENCH}_perf"
    BENCH_DIR="$RESULTS_DIR/$BENCH"
    
    echo ""
    echo "=============================================="
    echo "Running: $BENCH"
    echo "=============================================="
    
    if [[ ! -f "$BENCH_EXE" ]]; then
        echo "  SKIPPED: $BENCH_EXE not found"
        echo "$BENCH: SKIPPED (executable not found)" >> "$SUMMARY_FILE"
        continue
    fi
    
    mkdir -p "$BENCH_DIR"
    
    # Clean any previous run data
    rm -rf rocp rank_*_timestamps.csv
    
    # Run benchmark
    echo "  Running benchmark..."
    if "$MPI_DIR/bin/mpirun" -np "$NP" \
        "$ROCPROFV3" --kernel-trace -f parquet -d rocp -- \
        "$BENCH_EXE" $BENCH_ARGS > "$BENCH_DIR/benchmark_output.txt" 2>&1; then
        
        echo "  Benchmark completed successfully"
        
        # Copy timestamp files
        cp rank_*_timestamps.csv "$BENCH_DIR/" 2>/dev/null || true
        
        # Find parquet directory
        PARQUET_SUBDIR=$(ls -d rocp/*/ 2>/dev/null | head -1)
        
        if [[ -n "$PARQUET_SUBDIR" ]]; then
            # Copy parquet data
            cp -r "$PARQUET_SUBDIR" "$BENCH_DIR/parquet_data/"
            
            # Run analysis
            echo "  Running analysis..."
            python3 "$SCRIPT_DIR/analyze_rccl.py" \
                -t "$BENCH_DIR/rank_*_timestamps.csv" \
                -p "$BENCH_DIR/parquet_data" \
                -o "$BENCH_DIR/analysis" > "$BENCH_DIR/analysis_log.txt" 2>&1 || true
            
            python3 "$SCRIPT_DIR/analyze_kernels.py" \
                -t "$BENCH_DIR/rank_*_timestamps.csv" \
                -p "$BENCH_DIR/parquet_data" \
                -o "$BENCH_DIR/analysis" >> "$BENCH_DIR/analysis_log.txt" 2>&1 || true
            
            # Extract key stats for summary
            if [[ -f "$BENCH_DIR/analysis/benchmark_statistics.csv" ]]; then
                # Get average bandwidth for large messages
                AVG_BW=$(python3 -c "
import polars as pl
df = pl.read_csv('$BENCH_DIR/analysis/benchmark_statistics.csv')
large = df.filter(pl.col('size') >= 1048576)
if len(large) > 0:
    bw = (large['size'] / large['wall_time_us'] / 1000.0).mean()
    print(f'{bw:.2f}')
else:
    print('N/A')
" 2>/dev/null || echo "N/A")
                
                echo "$BENCH: SUCCESS - Avg BW (>=1MB): ${AVG_BW} GB/s" >> "$SUMMARY_FILE"
                echo "  Average bandwidth (>=1MB): ${AVG_BW} GB/s"
            else
                echo "$BENCH: SUCCESS (no stats)" >> "$SUMMARY_FILE"
            fi
        else
            echo "$BENCH: SUCCESS (no parquet data)" >> "$SUMMARY_FILE"
        fi
    else
        echo "  Benchmark FAILED"
        echo "$BENCH: FAILED" >> "$SUMMARY_FILE"
    fi
    
    # Clean up
    rm -rf rocp rank_*_timestamps.csv
done

echo ""
echo "=============================================="
echo "ALL BENCHMARKS COMPLETE"
echo "=============================================="
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
cat "$SUMMARY_FILE"
echo ""
echo "Results directory contents:"
ls -la "$RESULTS_DIR"

