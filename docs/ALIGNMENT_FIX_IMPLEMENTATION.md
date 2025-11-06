# Alignment Issue Fix Implementation

## Overview

Modified `run_timing_sweep.py` to automatically adjust minimum message sizes for benchmarks affected by the 16-byte alignment issue, preventing zero-size outputs.

## Implementation Date
November 4, 2025

## Problem Summary

Six RCCL benchmarks (`all_gather`, `gather`, `scatter`, `reduce_scatter`, `alltoall`, `hypercube`) use an aggressive 16-byte alignment mask that causes small message sizes to be zeroed out when divided across multiple MPI ranks.

**Formula causing issue:**
```c
paramcount = (count/nranks) & -(16/eltSize)
```

## Solution

Added automatic minimum size calculation to `run_timing_sweep.py` that:
1. Detects affected benchmarks
2. Calculates safe minimum size: `min_size = nranks * 16` (rounded to next power of 2)
3. Adjusts sweep range to start from safe minimum
4. Reports adjustment to user

## Code Changes

### New Function: `calculate_min_size_for_benchmark()`

```python
def calculate_min_size_for_benchmark(benchmark_name, num_ranks, datatype='float'):
    """
    Calculate minimum message size to avoid zero-size outputs due to alignment.
    
    Affected benchmarks use alignment mask -(16/eltSize) which can zero out
    small per-rank message sizes. Formula: (count/nranks) & -(16/eltSize)
    
    To avoid zero output, we need: (count/nranks) >= (16/eltSize)
    Therefore: count >= nranks * (16/eltSize)
    Therefore: size_bytes >= nranks * 16
    """
    # Benchmarks affected by 16-byte alignment issue
    affected_benchmarks = [
        'all_gather', 'gather', 'scatter', 
        'reduce_scatter', 'alltoall', 'hypercube'
    ]
    
    if benchmark_name not in affected_benchmarks:
        # Safe benchmarks can start at 8 bytes
        return 8
    
    # For affected benchmarks, minimum size is nranks * 16 bytes
    min_size = num_ranks * 16
    
    # Round up to next power of 2 if not already
    power = 1
    while power < min_size:
        power *= 2
    
    return power
```

### Modified Function: `generate_size_sweep()`

Changed signature to accept `min_size` parameter:
```python
def generate_size_sweep(min_size=8):
    """Generate a power-of-2 sweep of message sizes from min_size to 1 GiB"""
    # ... implementation starts from min_size instead of hardcoded 8
```

### Modified Function: `run_benchmark_for_full_range()`

Added `min_size` parameter and uses it for `-b` (begin) argument to benchmark.

### Modified: `main()`

Added minimum size calculation and user notification:
```python
# Calculate minimum size based on benchmark and rank count
num_ranks = args.ranks if args.mpi else 1
min_size = calculate_min_size_for_benchmark(args.benchmark, num_ranks, args.datatype)

# Generate size sweep starting from calculated minimum
sizes = generate_size_sweep(min_size)

# Report if minimum size was adjusted
if min_size > 8:
    print(f"Note: Minimum size adjusted to {min_size} bytes to avoid alignment issues")
    print(f"      (Benchmark '{args.benchmark}' with {num_ranks} ranks)")
```

## Minimum Size Table

For affected benchmarks with float data type (4 bytes):

| Ranks | Min Size Formula | Actual Min Size | Skipped Sizes |
|-------|-----------------|-----------------|---------------|
| 1     | 1 × 16 = 16     | 16 bytes        | 8             |
| 2     | 2 × 16 = 32     | 32 bytes        | 8, 16         |
| 4     | 4 × 16 = 64     | 64 bytes        | 8, 16, 32     |
| 8     | 8 × 16 = 128    | 128 bytes       | 8, 16, 32, 64 |

For safe benchmarks (all_reduce, broadcast, reduce, sendrecv, all_reduce_bias):
- Always starts at 8 bytes regardless of rank count

## Verification Tests

### Test 1: alltoall with 2 ranks
```bash
python3 run_timing_sweep.py alltoall --datatype float --ranks 2 --mpi
```
**Expected:** Start at 32 bytes  
**Result:** ✓ Starts at 32 bytes, no zero-size entries

### Test 2: alltoall with 8 ranks
```bash
python3 run_timing_sweep.py alltoall --datatype float --ranks 8 --mpi
```
**Expected:** Start at 128 bytes  
**Result:** ✓ Starts at 128 bytes, no zero-size entries

### Test 3: all_reduce with 8 ranks (safe benchmark)
```bash
python3 run_timing_sweep.py all_reduce --datatype float --ranks 8 --mpi
```
**Expected:** Start at 8 bytes  
**Result:** ✓ Starts at 8 bytes, no adjustment message

## User Experience

### Before Fix
```
Size sweep: 28 sizes from 8 to 1073741824 bytes
...
Benchmark output shows:
   0    0    float    none    -1    15.02    0.00    0.00    ...
   0    0    float    none    -1    15.31    0.00    0.00    ...
  32    4    float    none    -1    35.91    0.00    0.00    ...
```

### After Fix
```
Note: Minimum size adjusted to 32 bytes to avoid alignment issues
      (Benchmark 'alltoall' with 2 ranks)
Size sweep: 26 sizes from 32 to 1073741824 bytes
...
Benchmark output shows:
  32    4    float    none    -1    37.49    0.00    0.00    ...
  64    8    float    none    -1    34.59    0.00    0.00    ...
 128   16    float    none    -1    34.51    0.00    0.00    ...
```

## Benefits

1. **Automatic Protection:** Users don't need to manually calculate safe minimum sizes
2. **No Zero-Size Entries:** Eliminates spurious zero-size benchmark results
3. **Complete Data:** Ensures all tested sizes produce valid measurements
4. **Transparent:** Clearly reports when and why adjustments are made
5. **Backward Compatible:** Safe benchmarks continue to work as before
6. **Analysis-Ready:** Output data is immediately usable by analysis scripts

## Limitations

### Data Loss for Small Sizes

Affected benchmarks with multiple ranks will skip small message sizes:
- **2 ranks:** Lose 8, 16 byte measurements
- **8 ranks:** Lose 8, 16, 32, 64 byte measurements

**Workaround:** Run affected benchmarks with 1 rank to capture small-size data (though this doesn't test multi-rank behavior).

### Root Cause Not Fixed

This is a **workaround** in the test harness, not a fix to the benchmark code itself. The underlying alignment issue in `src/all_gather.cu`, `src/alltoall.cu`, etc. still exists.

**To fully fix:** Modify the `GetCollByteCount` functions in the affected benchmark source files (see `docs/ALIGNMENT_ISSUE_ANALYSIS.md` for details).

## Related Documentation

- **`docs/ALIGNMENT_ISSUE_ANALYSIS.md`** - Comprehensive root cause analysis
- **`docs/to-do.md`** - Tracking remaining work

## Future Work

1. Consider fixing the root cause in benchmark source files
2. Add command-line option to override minimum size if needed
3. Extend to handle other data types (currently assumes worst case)
4. Add validation to detect if zero-size entries still occur

## Testing Recommendations

When using affected benchmarks:
1. Check script output for "Minimum size adjusted" message
2. Verify first benchmark output line shows expected minimum size
3. Confirm no zero-size entries in `*_benchmark_output.txt`
4. Verify timing CSV files have matching non-zero `size_bytes` values

## Script Version

This fix is included in `run_timing_sweep.py` version 4.0+



