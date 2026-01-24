#!/bin/bash
# Run variance analysis on all benchmark results
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="/work/lmeadows/rccl/rccl-tests/benchmark_results/20251212_120646"

echo "Running variance analysis on all benchmarks..."
echo ""

for BENCH_DIR in "$RESULTS_DIR"/*/; do
    BENCH_NAME=$(basename "$BENCH_DIR")
    
    # Skip if no parquet data
    if [[ ! -d "$BENCH_DIR/parquet_data" ]]; then
        echo "Skipping $BENCH_NAME (no parquet data)"
        continue
    fi
    
    # Check for timestamp files
    if ! ls "$BENCH_DIR"/rank_*_timestamps.csv >/dev/null 2>&1; then
        echo "Skipping $BENCH_NAME (no timestamp files)"
        continue
    fi
    
    echo "Analyzing: $BENCH_NAME"
    
    python3 "$SCRIPT_DIR/analyze_variance.py" \
        -t "$BENCH_DIR/rank_*_timestamps.csv" \
        -p "$BENCH_DIR/parquet_data" \
        -o "$BENCH_DIR/analysis" 2>&1 | grep -E "^(Saved:|  |Extracting|Loading)"
    
    echo ""
done

echo "Variance analysis complete!"
echo ""
echo "Results added to each benchmark's analysis/ directory:"
ls "$RESULTS_DIR"/*/analysis/*variance* 2>/dev/null || echo "(listing files...)"

