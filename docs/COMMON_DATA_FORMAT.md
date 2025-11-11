# Common Data Format and Loading Functions

## Overview

The `scripts/common_data.py` module provides standardized data loading and processing functions for RCCL benchmark analysis. This ensures consistency across all analysis scripts and eliminates duplicate code.

## Key Changes

### 1. CSV Benchmark Output
- **Old**: Benchmarks only generated text output, parsed with regex
- **New**: Benchmarks generate CSV output using `-x <file>.csv -Z csv` flags
- **Added Column**: `busbwfactor` - the ratio of bus bandwidth to algorithm bandwidth

### 2. Centralized Functions
All data loading functions are now in `scripts/common_data.py`:
- `load_benchmark_output()` - Loads benchmark CSV
- `load_timing_data()` - Loads per-rank timing CSVs
- `load_kernel_traces()` - Loads ROCProfiler kernel traces
- `load_timestamp_ranges()` - Loads benchmark timestamp ranges
- `calculate_bus_bandwidth()` - Calculates bus bandwidth from size and time

### 3. Bus Bandwidth Factor
**Source of Truth**: The C++ benchmark code (`src/common.cu`)

The benchmark now calculates and exports `busBwFactor = busBw / algBw` in the CSV.

**Python Usage**:
- For **wall clock times**: Use `busbw_gbs` directly from CSV
- For **kernel timings**: Use `calculate_bus_bandwidth(size, time, collective_name, nranks)`

---

## DataFrame Schemas

### Benchmark Output DataFrame
Returned by: `load_benchmark_output(run_dir, benchmark_name)`

```python
pd.DataFrame with columns:
    - numCycle (int): Benchmark cycle number
    - collective (str): Collective operation name (e.g., 'AllReduce')
    - ranks (int): Number of MPI ranks
    - rankspernode (int): Ranks per node
    - gpusperrank (int): GPUs per rank
    - size_bytes (int): Message size in bytes
    - data_type (str): Data type (e.g., 'float')
    - redop (str): Reduction operation (e.g., 'sum')
    - inplace (int): 0 for out-of-place, 1 for in-place
    - wall_time_us (float): Wall clock time in microseconds
    - algbw_gbs (float): Algorithm bandwidth in GB/s (G=10^9)
    - busbw_gbs (float): Bus bandwidth in GB/s (G=10^9)
    - busbw_factor (float): Bus bandwidth factor (busbw / algbw)
    - errors (int): Number of errors
```

**Example**:
```python
from common_data import load_benchmark_output

df = load_benchmark_output('/path/to/run_all_reduce_20251110_135024')
print(df[['size_bytes', 'inplace', 'wall_time_us', 'busbw_gbs']].head())
```

### Timing Data DataFrame
Returned by: `load_timing_data(run_dir)`

```python
pd.DataFrame with columns:
    - rank (int): MPI rank
    - size_bytes (int): Message size
    - inplace (int): 0 for out-of-place, 1 for in-place
    - iteration (int): Iteration number
    - kernel_time_us (float): Individual kernel execution time
    - IQR_filtered (bool): Whether included after IQR filtering
```

### Kernel Traces DataFrame
Returned by: `load_kernel_traces(run_dir, rank_pid_map)`

```python
pd.DataFrame with columns:
    - rank (int): MPI rank
    - pid (int): Process ID
    - kernel_name (str): Kernel function name
    - begin_ns (int): Kernel start time in nanoseconds
    - end_ns (int): Kernel end time in nanoseconds
    - duration_ns (int): Kernel duration in nanoseconds
    - is_nccl (bool): Whether kernel is an NCCL kernel
```

### Timestamp Ranges DataFrame
Returned by: `load_timestamp_ranges(run_dir, ranks)`

```python
pd.DataFrame with columns:
    - rank (int): MPI rank
    - size_bytes (int): Message size
    - operation_mode (str): 'oop' or 'inp'
    - start_ns (int): Range start time in nanoseconds
    - end_ns (int): Range end time in nanoseconds
    - duration_ns (int): Range duration in nanoseconds
    - config (str): Configuration string
```

---

## Bandwidth Calculations

### Algorithm Bandwidth
```python
algbw_gbs = size_bytes / time_us / 1000.0
```
- Represents effective bandwidth from application's perspective
- "I moved X bytes in Y time"

### Bus Bandwidth
```python
busbw_gbs = algbw_gbs * busbw_factor
```
- Represents actual network utilization per rank
- Accounts for communication pattern (send + receive)

**Factors by Collective**:
| Collective | Factor | Formula |
|-----------|--------|---------|
| AllReduce | 2(N-1)/N | Ring: 2 phases × (N-1)/N |
| ReduceScatter | (N-1)/N | Single phase |
| AllGather | (N-1)/N | Single phase |
| Reduce | 1.0 | No amplification |
| Broadcast | 1.0 | No amplification |
| AlltoAll | (N-1)/N | Peer-to-peer |

**Example for AllReduce with 8 ranks**:
```
factor = 2 * (8-1) / 8 = 1.75
busbw = algbw * 1.75
```

### Using Bandwidth Functions

**From common_data module**:
```python
from common_data import calculate_algorithm_bandwidth, calculate_bus_bandwidth

# Algorithm bandwidth
algbw = calculate_algorithm_bandwidth(size_bytes=1024, time_us=50.0)
# Result: 20.48 GB/s

# Bus bandwidth (for kernel timings)
busbw = calculate_bus_bandwidth(
    size_bytes=1024,
    time_us=50.0, 
    collective_name='all_reduce',
    nranks=8
)
# Result: 35.84 GB/s (20.48 * 1.75)
```

---

## Migration Guide

### Old Code (Text Parsing)
```python
# Old way - regex parsing of text output
output_file = os.path.join(run_dir, f'{benchmark_name}_benchmark_output.txt')
wall_times = []

with open(output_file, 'r') as f:
    for line in f:
        match = re.match(r'^\s*(\d+)\s+.*\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
        if match:
            size = int(match.group(1))
            time = float(match.group(2))
            algbw = float(match.group(3))
            busbw = float(match.group(4))
            wall_times.append({'size_bytes': size, 'wall_time_us': time})

df = pd.DataFrame(wall_times)
```

### New Code (CSV Loading)
```python
# New way - direct CSV loading
from common_data import load_benchmark_output

df = load_benchmark_output(run_dir, benchmark_name)
# df already has size_bytes, wall_time_us, algbw_gbs, busbw_gbs, busbw_factor
```

### Calculating Bandwidth

**Old Code**:
```python
# Manual calculation (algorithm bandwidth only)
bw = size_bytes / time_us / 1000
```

**New Code**:
```python
from common_data import calculate_bus_bandwidth

# For kernel timings (not in CSV)
bw = calculate_bus_bandwidth(size_bytes, time_us, benchmark_name, nranks)

# For wall clock times (in CSV)
bw = df['busbw_gbs']  # Already calculated by benchmark!
```

---

## Scripts Updated

All analysis scripts now use `common_data.py`:

1. ✅ `plot_size_vs_time_plotly.py` - Interactive performance plots
2. ✅ `analyze_timing_stats.py` - Statistical analysis
3. ✅ `segment_performance_bic.py` - BIC segmentation
4. ✅ `create_boxplots.py` - Timing distribution boxplots
5. ✅ `plot_kernel_timeline.py` - Kernel timeline visualization

---

## Benefits

### 1. Correctness
- **Single source of truth** for bandwidth factors (C++ benchmark)
- Consistent data loading across all scripts
- No more regex parsing errors

### 2. Maintainability
- Fix bugs in one place → all scripts benefit
- Clear DataFrame schemas reduce confusion
- Easy to add new data sources

### 3. Performance
- CSV parsing is faster than regex
- pandas optimizations for large datasets
- No duplicate file reading

### 4. Usability
- Auto-detect benchmark name from directory
- Graceful handling of missing files
- Consistent column naming

---

## File Locations

```
rccl-tests/
├── src/
│   └── common.cu                    # Exports busbwfactor in CSV
├── scripts/
│   ├── common_data.py               # ⭐ Central data loading module
│   ├── run_timing_sweep.py          # Generates CSV with -x and -Z flags
│   ├── plot_size_vs_time_plotly.py  # Uses common_data
│   ├── analyze_timing_stats.py      # Uses common_data
│   ├── segment_performance_bic.py   # Uses common_data
│   └── ...
└── docs/
    └── COMMON_DATA_FORMAT.md        # This file
```

---

## Example: Complete Analysis Workflow

```python
from common_data import (
    load_benchmark_output,
    load_timing_data,
    load_bic_segmentation,
    calculate_bus_bandwidth
)

# 1. Load benchmark data
run_dir = '/path/to/run_all_reduce_20251110_135024'
benchmark_df = load_benchmark_output(run_dir)
timing_df = load_timing_data(run_dir)

# 2. Get wall clock bus bandwidth (from CSV)
wall_busbw = benchmark_df[benchmark_df['inplace'] == 0]['busbw_gbs']

# 3. Calculate kernel bus bandwidth
nranks = benchmark_df['ranks'].iloc[0]
kernel_busbw = calculate_bus_bandwidth(
    size_bytes=timing_df['size_bytes'].values,
    time_us=timing_df['kernel_time_us'].values,
    collective_name='all_reduce',
    nranks=nranks
)

# 4. Load segmentation results
segmentation = load_bic_segmentation(run_dir)
print(f"Found {segmentation['n_segments']} performance segments")
```

---

## Testing

After generating new benchmark data with CSV output:

```bash
# Run a benchmark
python3 scripts/run_timing_sweep.py all_reduce

# Verify CSV was created
ls data/*/run_all_reduce_*/all_reduce_benchmark_output.csv

# Test common_data functions
python3 -c "
from scripts.common_data import load_benchmark_output
df = load_benchmark_output('data/.../run_all_reduce_...')
print(df.columns)
print(df[['size_bytes', 'busbw_gbs', 'busbw_factor']].head())
"
```

---

## Future Enhancements

1. **Caching**: Cache loaded DataFrames to parquet for faster re-analysis
2. **Validation**: Add schema validation for CSV files
3. **Metadata**: Include more run metadata (hostname, GPU model, RCCL version)
4. **Compression**: Support gzip-compressed CSV files

---

## Questions?

See also:
- `scripts/common_data.py` - Source code with detailed docstrings
- `src/common.cu` - C++ benchmark code that generates CSV
- `docs/KERNEL_TIMELINE_DATAFRAME_REFACTOR.md` - Related DataFrame refactoring

