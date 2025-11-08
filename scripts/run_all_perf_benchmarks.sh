#!/bin/bash
#
# Run pipeline on all *_perf benchmarks
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BENCHMARKS=(
    "all_reduce"
    "all_gather"
    "broadcast"
    "reduce"
    "reduce_scatter"
    "scatter"
    "gather"
    "alltoall"
    "alltoallv"
    "hypercube"
    "sendrecv"
    "all_reduce_bias"
)

echo "========================================================================"
echo "Running pipeline on ${#BENCHMARKS[@]} benchmarks"
echo "========================================================================"
echo

SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_BENCHMARKS=()

for benchmark in "${BENCHMARKS[@]}"; do
    echo "========================================================================"
    echo "Starting: $benchmark"
    echo "========================================================================"
    
    if python3 run_full_pipeline.py "$benchmark"; then
        echo "✅ $benchmark completed successfully"
        ((SUCCESS_COUNT++))
    else
        echo "❌ $benchmark failed"
        ((FAIL_COUNT++))
        FAILED_BENCHMARKS+=("$benchmark")
    fi
    
    echo
    echo "Progress: $((SUCCESS_COUNT + FAIL_COUNT)) / ${#BENCHMARKS[@]} completed"
    echo
done

echo "========================================================================"
echo "Final Summary"
echo "========================================================================"
echo "Total benchmarks: ${#BENCHMARKS[@]}"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"

if [ $FAIL_COUNT -gt 0 ]; then
    echo
    echo "Failed benchmarks:"
    for benchmark in "${FAILED_BENCHMARKS[@]}"; do
        echo "  - $benchmark"
    done
fi

echo "========================================================================"

exit $FAIL_COUNT

