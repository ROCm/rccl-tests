# Benchmark Code: Skip Kernel Execution for Invalid Sizes

## Issue

The RCCL benchmark binaries were executing kernels for invalid message sizes (due to alignment constraints), even though `run_timing_sweep.py` calculated the correct minimum size. This caused:

1. **Kernel data recorded for invalid sizes** (16, 32, 64 bytes for scatter/gather with 8 ranks)
2. **Alignment shift in plots**: Kernel timing data included 3 extra sizes not in benchmark output
3. **Inefficiency**: Wasted GPU cycles executing kernels that produce zero-byte messages

## Root Cause

The benchmark loop in `src/common.cu` was:
1. Calling `setupArgs()` for each size
2. Printing the output row header
3. Calling `BenchTime()` to execute kernels **regardless** of whether the computed message size was zero

The check `std::max(args->sendBytes, args->expectedBytes)` was only used for **printing** and **reporting**, not for deciding whether to execute kernels.

## Solution

Modified the main benchmark loop in `src/common.cu` (function `TimeTest`) to:

### 1. Check for Invalid Sizes

After `setupArgs()` is called, compute:
```cpp
auto largestMessageSize = std::max(args->sendBytes, args->expectedBytes);
```

### 2. Add Debug Logging

When an invalid size is detected (rank 0 only):
```cpp
if (largestMessageSize == 0 && args->proc == 0) {
  fprintf(stderr, "# DEBUG: Skipping size=%lu (sendBytes=%lu, expectedBytes=%lu, largestMessageSize=0)\n", 
          size, args->sendBytes, args->expectedBytes);
  fflush(stderr);
}
```

### 3. Skip Kernel Execution

If `largestMessageSize == 0`:
- Print placeholder values: `"SKIP"`, `"0.00"`, `"0.00"`, `"N/A"`
- **Skip** both `BenchTime()` calls (out-of-place and in-place)
- Use `continue` to jump to next size in the loop

### 4. Normal Execution for Valid Sizes

Only execute kernels when `largestMessageSize > 0`

## Code Changes

**File:** `src/common.cu` (lines 1009-1058)

**Before:**
```cpp
for (size_t size = args->minbytes; size<=args->maxbytes; size = ...) {
  setupArgs(size, type, args);
  PRINT("%12li  ...", std::max(args->sendBytes, args->expectedBytes), ...);
  if (enable_out_of_place) {
    TESTCHECK(BenchTime(args, type, op, root, 0));  // Always called!
    ...
  }
  if (enable_in_place)
    TESTCHECK(BenchTime(args, type, op, root, 1));  // Always called!
  ...
}
```

**After:**
```cpp
for (size_t size = args->minbytes; size<=args->maxbytes; size = ...) {
  setupArgs(size, type, args);
  
  // Check if this size produces valid message sizes
  auto largestMessageSize = std::max(args->sendBytes, args->expectedBytes);
  
  // Debug logging for invalid sizes
  if (largestMessageSize == 0 && args->proc == 0) {
    fprintf(stderr, "# DEBUG: Skipping size=%lu (sendBytes=%lu, expectedBytes=%lu, largestMessageSize=0)\n", 
            size, args->sendBytes, args->expectedBytes);
    fflush(stderr);
  }
  
  PRINT("%12li  ...", largestMessageSize, ...);
  
  // Skip kernel execution if the size would produce zero-byte messages
  if (largestMessageSize == 0) {
    PRINT("  %7s  %6s  %6s  %5s", "SKIP", "0.00", "0.00", "N/A");
    if(output_algo_proto_channels) {
      PRINT("%8s  %8s  %10s", "N/A", "N/A", "N/A");
    }
    PRINT("\n");
    continue;  // Skip to next size
  }
  
  // Run benchmark normally for valid sizes
  if (enable_out_of_place) {
    TESTCHECK(BenchTime(args, type, op, root, 0));
    ...
  }
  if (enable_in_place)
    TESTCHECK(BenchTime(args, type, op, root, 1));
  ...
}
```

## Benefits

### 1. **No Kernel Execution for Invalid Sizes**
- ROCProfiler won't record kernel data for sizes that produce zero-byte messages
- Eliminates wasted GPU cycles

### 2. **Clean Kernel Timing Data**
- CSV files (`all_rank*.csv`) will only contain timing data for valid sizes
- No more misalignment between kernel data and benchmark output

### 3. **Eliminates Need for Python Filtering**
- The fix in `plot_size_vs_time_plotly.py` and `create_boxplots.py` is still useful as a safeguard
- But now the data is clean at the source

### 4. **Clear Debugging**
- Debug messages to stderr show exactly which sizes are being skipped and why
- Makes alignment issues immediately visible during benchmark runs

### 5. **Consistent Behavior**
- Benchmark output and kernel timing data are now perfectly aligned
- Applies to all affected benchmarks: scatter, gather, reduce_scatter, alltoall, etc.

## Example Output

For `scatter_perf` with 8 ranks and minBytes=8:

**Stderr (debug log):**
```
# DEBUG: Skipping size=8 (sendBytes=8, expectedBytes=0, largestMessageSize=0)
# DEBUG: Skipping size=16 (sendBytes=16, expectedBytes=0, largestMessageSize=0)
# DEBUG: Skipping size=32 (sendBytes=32, expectedBytes=0, largestMessageSize=0)
```

**Stdout (benchmark output):**
```
#       size         count    type   redop    root     time   algbw   busbw
           0             0    float    none       0     SKIP    0.00    0.00  N/A
           0             0    float    none       0     SKIP    0.00    0.00  N/A
           0             0    float    none       0     SKIP    0.00    0.00  N/A
          64            16    float    none       0   123.45   12.34   23.45  N/A
         128            32    float    none       0   234.56   23.45   34.56  N/A
```

## Rebuild Instructions

After modifying `src/common.cu`:

```bash
cd /work/lmeadows/rccl/rccl-tests
source scripts/sourceme
./domake
```

All benchmark binaries will be rebuilt with the fix.

## Testing

To verify the fix is working:

1. **Check for debug messages:**
   ```bash
   mpirun -np 8 ./build/scatter_perf -b 8 -e 1G 2>&1 | grep DEBUG
   ```

2. **Verify "SKIP" in output:**
   ```bash
   mpirun -np 8 ./build/scatter_perf -b 8 -e 1G 2>&1 | grep SKIP
   ```

3. **Run full pipeline and check CSV files:**
   ```bash
   python3 scripts/run_timing_sweep.py scatter --ranks 8
   # Check that all_rank*.csv doesn't have entries for sizes 16, 32, 64
   head -20 run_scatter_*/all_rank0.csv
   ```

## Related Fixes

This fix complements the Python-side filtering added in:
- `plot_size_vs_time_plotly.py` - filters kernel data against valid benchmark sizes
- `create_boxplots.py` - filters kernel data against valid benchmark sizes

Together, these provide defense-in-depth:
- **C++ fix**: Prevents invalid data from being generated
- **Python fix**: Handles any edge cases or legacy data

## Update: Fixed Timestamp Size Labeling

**Issue:** Timestamps were recording per-rank size (`args->nbytes`) instead of total input size, causing misalignment in plots.

**Solution:**
1. Added `inputSize` field to `struct threadArgs` in `src/common.h`
2. Updated `setupArgs()` to store: `args->inputSize = size`
3. Updated timestamp print to use: `args->inputSize` instead of `args->nbytes`

**Result:** Timestamps and CSV files now use total input sizes that match benchmark output exactly.

**Example (scatter with 8 ranks):**
- **Before:** `Scatter_size_16` (per-rank size from input 128)
- **After:** `Scatter_size_128` (total input size) ✓

This ensures perfect alignment between kernel timing data and benchmark output sizes for plotting.

## Date
November 8, 2025 (updated same day)

