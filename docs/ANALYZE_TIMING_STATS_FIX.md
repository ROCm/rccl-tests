# analyze_timing_stats.py Format String Fix

## Issue

The `analyze_timing_stats.py` script was printing extraneous format codes instead of formatted data after the "Individual kernel timings aggregated across all ranks" banner.

### Observed Output (Before Fix)

```
Individual kernel timings aggregated across all ranks
====================================================================================================

IN-PLACE Operations:
--------------------------------------------------------------------------------
7
-
6.1f
6.1f
6.1f
...
```

### Root Cause

Two functions had incomplete/broken format strings:

1. **`format_size()` function** (lines 189-198):
   - Was returning format codes like `"6.1f"` instead of formatted strings
   - Should have been returning formatted size strings like `"128 MiB"`

2. **`print_summary_table()` function** (lines 200-246):
   - Was assigning format codes to variables: `header = "7"`, `count_str = "4d"`, `line = "6.1f"`
   - Was using wrong column names: `row['count']` instead of `row['kernel_count']`
   - Should have been using f-strings to format actual data values

## Fix Applied

### 1. Fixed `format_size()` function

**Before:**
```python
def format_size(size_bytes):
    """Format size in human readable format"""
    if size_bytes >= 1024**3:
        return "6.1f"
    elif size_bytes >= 1024**2:
        return "6.1f"
    elif size_bytes >= 1024:
        return "6.1f"
    else:
        return "6.0f"
```

**After:**
```python
def format_size(size_bytes):
    """Format size in human readable format"""
    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.1f} GiB"
    elif size_bytes >= 1024**2:
        return f"{size_bytes / (1024**2):.0f} MiB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.0f} KiB"
    else:
        return f"{size_bytes} B"
```

### 2. Fixed `print_summary_table()` function

**Key changes:**
- Created proper header string with column names
- Used correct column names from DataFrame: `kernel_count`, `kernel_mean_us`, etc.
- Formatted all values using f-strings with actual data

**Before:**
```python
header = "7"
print(header)
# ...
count_str = "4d"
line = "6.1f"
print(line)
```

**After:**
```python
header = f"{'Size':>12} {'Count':>7} {'Wall':>8}  {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'P25':>8} {'P75':>8} {'CV%':>6}"
print(header)
# ...
count_str = f"{int(row['kernel_count']):4d}"
line = (f"{size_str:>12} {count_str:>7} {wall_str:>8}  "
       f"{row['kernel_mean_us']:8.1f} {row['kernel_std_us']:8.1f} "
       f"{row['kernel_min_us']:8.1f} {row['kernel_max_us']:8.1f} "
       f"{row['kernel_p25_us']:8.1f} {row['kernel_p75_us']:8.1f} "
       f"{row['kernel_cv_percent']:6.1f}")
print(line)
```

## Expected Output (After Fix)

```
====================================================================================================
RCCL ALL_REDUCE - Statistical Timing Summary
====================================================================================================
Individual kernel timings aggregated across all ranks
====================================================================================================

IN-PLACE Operations:
--------------------------------------------------------------------------------
        Size   Count     Wall      Mean      Std      Min      Max      P25      P75    CV%
-------------------------------------------------------------------------------------------
         8 B     800    25.9       16.2      4.0     13.5     46.1     15.3     15.9   24.7
        16 B     800    21.0       15.6      1.0     14.0     29.8     15.2     15.9    6.1
        32 B     800    21.8       16.5      0.8     14.8     26.5     16.2     16.8    4.9
...
```

## Verification

Tested with:
```bash
python3 scripts/analyze_timing_stats.py /work/lmeadows/rccl/data/banff-ccs-aus-g01-14/run_all_reduce_20251108_063035
```

Results: ✅ Clean, properly formatted table with all data values displayed correctly.

## Date
November 8, 2025

