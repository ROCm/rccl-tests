# ROCTX Integration for RCCL Benchmarks

## Overview

ROCTX (ROCm Tracer eXtensions) annotations have been added to the RCCL benchmark suite to enable detailed profiling with ROCProfiler. This allows you to correlate GPU kernel launches with specific benchmark iterations and message sizes without modifying RCCL itself.

## What Was Added

### Code Changes

**File: `src/common.cu`**

1. **Header Include** (line 31-33):
   ```cpp
   #ifdef USE_ROCPROFILER
   #include <roctracer/roctx.h>
   #endif
   ```

2. **Benchmark Range Marker** (lines 682-694):
   - Wraps the entire `BenchTime()` function execution
   - Creates a hierarchical range with benchmark name, size, configuration
   - Format: `{benchmark}_size_{bytes}_{inplace/oop}_type_{type}_op_{op}_root_{root}`
   - Example: `reduce_scatter_size_1048576_oop_type_float_op_sum_root_0`

3. **Iteration Range Markers** (lines 714-718, 726-728):
   - Nested within the benchmark range
   - Marks each individual iteration of the benchmark loop
   - Format: `iter_{iteration_number}` (e.g., `iter_0000`, `iter_0001`, ...)
   - Allows correlation of specific kernel launches to iterations

4. **Range Cleanup** (lines 860-862):
   - Properly closes the benchmark range at the end of `BenchTime()`

### Build System Changes

**File: `src/Makefile`**

1. **Build Flag** (line 34):
   ```makefile
   USE_ROCPROFILER ?= 0
   ```

2. **Compiler/Linker Flags** (lines 134-137):
   ```makefile
   ifeq ($(USE_ROCPROFILER), 1)
   HIPCUFLAGS += -DUSE_ROCPROFILER -I$(ROCM_PATH)/include
   HIPLDFLAGS += -L$(ROCM_PATH)/lib -lroctx64
   endif
   ```

## Building with ROCTX Support

### Standard Build (without ROCTX)
```bash
cd /work/lmeadows/rccl/rccl-tests/src
make MPI=1 clean
make MPI=1
```

### Build with ROCTX Annotations
```bash
cd /work/lmeadows/rccl/rccl-tests/src
make MPI=1 USE_ROCPROFILER=1 clean
make MPI=1 USE_ROCPROFILER=1
```

The `USE_ROCPROFILER=1` flag:
- Defines the `USE_ROCPROFILER` preprocessor macro
- Includes ROCTX headers
- Links against `libroctx64.so`

## Running with ROCProfiler

### Basic Trace Collection

```bash
rocprof --roctx-trace \
        --hip-trace \
        --stats \
        --output-file=profile.csv \
        --timestamp on \
        mpirun -np 8 ./reduce_scatter_perf -b 128 -e 1G -f 2
```

### Trace Options Explained

- `--roctx-trace`: Enable ROCTX range tracking
- `--hip-trace`: Capture HIP API calls
- `--stats`: Generate summary statistics
- `--output-file=profile.csv`: Output file for trace data
- `--timestamp on`: Include timestamps in the trace

### Advanced Profiling with GPU Metrics

```bash
rocprof --roctx-trace \
        --hip-trace \
        --hsa-trace \
        --stats \
        --timestamp on \
        --basenames on \
        mpirun -np 8 ./reduce_scatter_perf -b 128 -e 1G -f 2
```

Additional options:
- `--hsa-trace`: Capture HSA (low-level GPU runtime) calls
- `--basenames on`: Use basename for output files (cleaner names)

### Collecting Hardware Counters

Create a metrics file `metrics.txt`:
```
pmc : SQ_WAVES Wavefronts
pmc : SQ_INSTS_VALU VALU Instructions
pmc : GRBM_GUI_ACTIVE GPU Busy %
```

Run with metrics:
```bash
rocprof --roctx-trace \
        --hip-trace \
        --stats \
        --timestamp on \
        --input metrics.txt \
        mpirun -np 8 ./reduce_scatter_perf -b 128 -e 1G -f 2
```

## Understanding the Output

### ROCTX Trace Format

ROCProfiler generates several output files:

1. **`results.json`**: Complete trace with all events
2. **`results.csv`**: Simplified CSV format
3. **`results.stats.csv`**: Summary statistics

### Example Trace Entry

```json
{
  "Name": "reduce_scatter_size_1048576_oop_type_float_op_sum_root_0/iter_0000",
  "pid": 12345,
  "tid": 1,
  "ts": 1000000,
  "dur": 150,
  "ph": "X",
  "args": {}
}
```

Fields:
- `Name`: Hierarchical range name (benchmark/iteration)
- `ts`: Start timestamp (microseconds)
- `dur`: Duration (microseconds)
- `ph`: Phase (`X` = complete event)

### Hierarchical Range Structure

```
reduce_scatter_size_1048576_oop_type_float_op_sum_root_0
├── iter_0000
│   └── [GPU kernels launched here]
├── iter_0001
│   └── [GPU kernels launched here]
├── iter_0002
│   └── [GPU kernels launched here]
...
```

## Analyzing Results

### Using Chrome Tracing

1. Convert to Chrome trace format:
   ```bash
   rocprof --chrome-trace results.json
   ```

2. Open `chrome://tracing` in Chrome/Chromium

3. Load the generated `.json` file

4. Navigate the timeline:
   - Zoom: W/S keys or mouse wheel
   - Pan: A/D keys or click-drag
   - Search: Ctrl+F to find specific ranges

### Extracting Per-Iteration Timings

Python script to parse ROCProfiler output:

```python
import json
import pandas as pd

# Load trace
with open('results.json', 'r') as f:
    trace = json.load(f)

# Extract ROCTX ranges
ranges = []
for event in trace['traceEvents']:
    if event.get('ph') == 'X' and 'Name' in event:
        name = event['Name']
        if 'iter_' in name:
            ranges.append({
                'name': name,
                'start_us': event['ts'],
                'duration_us': event['dur']
            })

df = pd.DataFrame(ranges)
print(df.describe())
```

### Correlating with Kernel Launches

ROCProfiler automatically associates GPU kernels with the active ROCTX range:

```python
# Find kernels within a specific iteration
iteration_range = "reduce_scatter_size_1048576_oop_type_float_op_sum_root_0/iter_0000"

kernels_in_iter = []
for event in trace['traceEvents']:
    if event.get('cat') == 'kernel' and event.get('args', {}).get('roctx_range') == iteration_range:
        kernels_in_iter.append({
            'kernel': event['Name'],
            'duration_us': event['dur']
        })

print(f"Kernels in {iteration_range}:")
for k in kernels_in_iter:
    print(f"  {k['kernel']}: {k['duration_us']} us")
```

## Performance Impact

### Overhead Analysis

ROCTX annotations have minimal overhead:
- **CPU overhead**: ~10-50 nanoseconds per `roctxRangePush`/`Pop` call
- **Memory overhead**: ~100 bytes per active range
- **GPU overhead**: None (annotations are CPU-side only)

### When to Use ROCTX

**Use ROCTX when:**
- Debugging performance issues
- Correlating timings with specific iterations
- Collecting detailed GPU metrics
- Profiling multi-rank behavior

**Don't use ROCTX when:**
- Running production benchmarks for publication
- Measuring absolute minimum latency
- Disk I/O is a bottleneck (trace files can be large)

## Integration with Existing Tools

### With `run_timing_sweep.py`

The benchmark sweep script can be modified to support ROCTX profiling:

```python
# In run_timing_sweep.py
def run_benchmark_with_profiling(benchmark_name, nranks, min_size, max_size):
    output_dir = create_output_directory(benchmark_name)
    
    rocprof_cmd = [
        "rocprof",
        "--roctx-trace",
        "--hip-trace",
        "--stats",
        f"--output-file={output_dir}/profile.csv",
        "--timestamp", "on"
    ]
    
    mpi_cmd = [
        "mpirun", "-np", str(nranks),
        f"./build/{benchmark_name}_perf",
        "-b", str(min_size),
        "-e", str(max_size),
        "-f", "2"
    ]
    
    full_cmd = rocprof_cmd + mpi_cmd
    subprocess.run(full_cmd, check=True)
```

### With Analysis Scripts

The segmentation and analysis scripts can be extended to use ROCProfiler data:

```python
# In segment_performance_bic.py
def load_rocprof_timings(profile_csv):
    """Load per-iteration timings from ROCProfiler trace."""
    df = pd.read_csv(profile_csv)
    
    # Filter for ROCTX ranges
    iter_ranges = df[df['Name'].str.contains('iter_')]
    
    # Extract size from range name
    iter_ranges['size'] = iter_ranges['Name'].apply(
        lambda x: int(x.split('_size_')[1].split('_')[0])
    )
    
    # Group by size and compute statistics
    stats = iter_ranges.groupby('size')['Duration'].agg(['mean', 'std', 'min', 'max'])
    
    return stats
```

## Troubleshooting

### ROCTX ranges not appearing in trace

**Problem**: ROCProfiler output doesn't show ROCTX ranges

**Solutions**:
1. Verify build: `strings ./build/reduce_scatter_perf | grep roctx`
   - Should show `roctxRangePush` if built correctly
2. Check ROCProfiler version: `rocprof --version`
   - Requires ROCm 4.0 or later
3. Ensure `--roctx-trace` flag is used

### Incomplete ranges (missing Pop)

**Problem**: Some ranges show as incomplete in the trace

**Solutions**:
1. Check for early exits in `BenchTime()` (errors, exceptions)
2. Verify all code paths call `roctxRangePop()`
3. Use `rocprof --flush-rate 1` to force immediate writes

### Large trace files

**Problem**: Trace files are too large (>1 GB)

**Solutions**:
1. Reduce benchmark iteration count: `-n 10` instead of default 20
2. Limit size range: `-b 1M -e 16M` instead of full sweep
3. Use `--basenames on` to reduce path lengths
4. Disable HIP trace: remove `--hip-trace` (keep only `--roctx-trace`)

### Multi-rank confusion

**Problem**: Can't distinguish which rank generated which trace

**Solutions**:
1. ROCProfiler creates separate files per rank: `results_<pid>.json`
2. Use MPI rank in output directory structure
3. Parse `pid` field in trace to correlate with MPI ranks

## Best Practices

1. **Build two versions**: One with ROCTX for profiling, one without for benchmarking
2. **Start small**: Profile a single size first, then expand to full sweep
3. **Use hierarchical ranges**: Nest ranges to create logical groupings
4. **Document range names**: Use descriptive, parseable naming conventions
5. **Automate analysis**: Write scripts to parse ROCProfiler output
6. **Archive traces**: Compress and store traces with benchmark metadata

## Example Workflow

```bash
# 1. Build with ROCTX support
cd /work/lmeadows/rccl/rccl-tests/src
make MPI=1 USE_ROCPROFILER=1 clean
make MPI=1 USE_ROCPROFILER=1

# 2. Run a focused profile
mkdir -p /work/lmeadows/rccl/profiles/reduce_scatter_debug
cd /work/lmeadows/rccl/profiles/reduce_scatter_debug

rocprof --roctx-trace \
        --hip-trace \
        --stats \
        --timestamp on \
        --basenames on \
        mpirun -np 8 /work/lmeadows/rccl/rccl-tests/build/reduce_scatter_perf \
               -b 1M -e 16M -f 2 -n 10

# 3. Analyze results
python3 /work/lmeadows/rccl/scripts/analyze_rocprof_trace.py results.json

# 4. View in Chrome
# Open chrome://tracing and load results.json

# 5. Rebuild without ROCTX for production benchmarks
cd /work/lmeadows/rccl/rccl-tests/src
make MPI=1 clean
make MPI=1
```

## References

- [ROCProfiler Documentation](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/)
- [ROCTX API Reference](https://rocm.docs.amd.com/projects/roctracer/en/latest/roctracer_api.html)
- [Chrome Tracing Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/)

