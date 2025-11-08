# Wall Clock Alignment Shift Fix

## Issue Description

In some benchmarks (scatter, gather, all_gather, reduce_scatter, alltoall, sendrecv), the Plotly plots showed wall clock times shifted right by 3 size units compared to the kernel timing data.

### Example: scatter with 8 ranks
- **Kernel data included:** 16, 32, 64, 128, 256, 512, ...
- **Benchmark output included:** 0, 128, 256, 512, ...
- **Result:** Wall clock markers appeared 3 sizes to the right

## Root Cause

### The Alignment Mask Issue

The affected benchmarks apply an alignment mask in their C++ code:

```cpp
// From src/scatter.cu line 13
*recvcount = (count/nranks) & -(16/eltSize);
```

For `float` (4 bytes) with 8 ranks:
- Size 16: `(16/8) & -4 = 2 & -4 = 0` ❌ (zeroed out)
- Size 32: `(32/8) & -4 = 4 & -4 = 4` ✅ (but still too small)
- Size 64: `(64/8) & -4 = 8 & -4 = 8` ✅ (but still problematic)
- Size 128: `(128/8) & -4 = 16 & -4 = 16` ✅ (first valid size)

### Why Benchmarks Run Extra Sizes

The `run_timing_sweep.py` script correctly calculates minimum sizes and passes them to the benchmark binary via `-b <min_size>`. However:

1. The benchmark binary has internal logic that may run additional sizes for testing/validation
2. Kernels are executed even for sizes that will produce invalid results
3. ROCProfiler records these kernel executions and timestamps
4. The benchmark output only includes sizes that passed validation

This creates a mismatch: kernel data includes all executed sizes, but benchmark output only includes valid sizes.

## Solution

Filter the kernel timing data to only include sizes present in the benchmark output.

### Changes Made

#### 1. plot_size_vs_time_plotly.py

Added filtering before building kernel summaries:

```python
def plot_interactive(run_dir, benchmark_name, timing_df, wall_df, segmentation):
    """Create interactive Plotly visualization."""
    
    # Get valid sizes from benchmark output (wall clock data)
    # This filters out sizes that were run but produced invalid results due to alignment
    valid_sizes = set()
    if len(wall_df) > 0:
        valid_sizes = set(wall_df['size_bytes'].unique())
        # Filter timing data to only include valid sizes
        if len(valid_sizes) > 0:
            timing_df = timing_df[timing_df['size_bytes'].isin(valid_sizes)].copy()
    
    # Build kernel summaries for OOP and INP
    kernel_df_oop = build_kernel_summary(timing_df, inplace_mode=0)
    kernel_df_inp = build_kernel_summary(timing_df, inplace_mode=1)
    # ...
```

#### 2. create_boxplots.py

Added helper function and filtering in main():

```python
def parse_benchmark_output(output_file):
    """
    Parse benchmark output to extract valid sizes.
    This filters out sizes that were executed but produced invalid results due to alignment.
    """
    valid_sizes = set()
    
    try:
        with open(output_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Match data lines - first column is size
                match = re.match(r'^\s*(\d+)', line)
                if match:
                    size_bytes = int(match.group(1))
                    valid_sizes.add(size_bytes)
    except Exception as e:
        print(f"Warning: Could not parse benchmark output: {e}")
    
    return valid_sizes

# In main():
output_file = os.path.join(args.run_dir, f'{benchmark_name}_benchmark_output.txt')
if os.path.exists(output_file):
    valid_sizes = parse_benchmark_output(output_file)
    if valid_sizes:
        before_count = len(timing_df)
        timing_df = timing_df[timing_df['size_bytes'].isin(valid_sizes)].copy()
        after_count = len(timing_df)
        if before_count != after_count:
            print(f"  Filtered out {before_count - after_count} measurements for invalid sizes")
```

### No Changes Needed

- **segment_performance_bic.py**: Already reads sizes directly from benchmark output, so only processes valid sizes
- **analyze_timing_stats.py**: Works with raw kernel data but doesn't compare against wall clock in a way that would show misalignment

## Verification

### Before Fix (scatter with 8 ranks)
```
Kernel data: 16, 32, 64, 128, 256, ...  (24 sizes)
Wall clock:  0, 128, 256, 512, ...      (25 sizes including 0)
Result: 3-size shift in plots
```

### After Fix (scatter with 8 ranks)
```
Loaded 38400 timing measurements
Filtered out 4800 measurements for invalid sizes
Size Range: 128 B to 128 MiB (21 sizes)
Result: Perfect alignment ✅
```

## Affected Benchmarks

The following benchmarks are affected by alignment issues (see `ALIGNMENT_ISSUE_ANALYSIS.md` for details):

1. `alltoall` - Line 14: `size = (size / nranks) & -(16/eltSize);`
2. `all_gather` - Line 14: `*recvcount = (count/nranks) & -(16/eltSize);`
3. `gather` - Line 13: `*recvcount = (count/nranks) & -(16/eltSize);`
4. `reduce_scatter` - Line 13: `*recvcount = (count/nranks) & -(16/eltSize);`
5. `scatter` - Line 13: `*recvcount = (count/nranks) & -(16/eltSize);`
6. `sendrecv` - Lines 27-28: Complex multi-GPU alignment

All of these now have the filter applied automatically in visualization scripts.

## Related Documentation

- `ALIGNMENT_ISSUE_ANALYSIS.md`: Detailed analysis of the alignment mask issue
- `ALIGNMENT_FIX_IMPLEMENTATION.md`: Implementation of automatic minimum size calculation
- `TIMING_MISMATCH_ROOT_CAUSE.txt`: Earlier investigation of similar CSV mismatch issues

## Date
November 8, 2025

