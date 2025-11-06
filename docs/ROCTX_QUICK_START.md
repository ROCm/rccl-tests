# ROCTX Quick Start Guide

## TL;DR

```bash
# Build with ROCTX
cd /work/lmeadows/rccl/rccl-tests/src
make MPI=1 USE_ROCPROFILER=1

# Profile a benchmark
rocprof --roctx-trace --hip-trace --stats --timestamp on \
  mpirun -np 8 ./build/reduce_scatter_perf -b 1M -e 16M -f 2

# View in Chrome
# Open chrome://tracing and load results.json
```

## What You Get

### Hierarchical Trace Ranges

```
reduce_scatter_size_1048576_oop_type_float_op_sum_root_0
  ├── iter_0000
  │     └── [GPU kernels]
  ├── iter_0001
  │     └── [GPU kernels]
  └── ...
```

### Range Name Format

```
{benchmark}_size_{bytes}_{inplace/oop}_type_{datatype}_op_{operation}_root_{root}
```

Examples:
- `reduce_scatter_size_1048576_oop_type_float_op_sum_root_0`
- `all_reduce_size_4194304_inplace_type_float_op_sum_root_0`
- `alltoall_size_8388608_oop_type_float_op_sum_root_0`

## Common Commands

### Basic Profiling
```bash
rocprof --roctx-trace --hip-trace --stats --timestamp on \
  mpirun -np 8 ./build/{benchmark}_perf -b {min} -e {max} -f 2
```

### With GPU Metrics
```bash
# Create metrics.txt:
# pmc : SQ_WAVES Wavefronts
# pmc : SQ_INSTS_VALU VALU Instructions

rocprof --roctx-trace --hip-trace --stats --timestamp on \
        --input metrics.txt \
  mpirun -np 8 ./build/{benchmark}_perf -b {min} -e {max} -f 2
```

### Focused Profile (Single Size)
```bash
rocprof --roctx-trace --hip-trace --stats --timestamp on \
  mpirun -np 8 ./build/reduce_scatter_perf -b 1M -e 1M -n 100
```

## Quick Analysis

### Extract Iteration Timings (Python)

```python
import json
import pandas as pd

with open('results.json') as f:
    trace = json.load(f)

ranges = [
    {
        'name': e['Name'],
        'start_us': e['ts'],
        'duration_us': e['dur']
    }
    for e in trace['traceEvents']
    if e.get('ph') == 'X' and 'iter_' in e.get('Name', '')
]

df = pd.DataFrame(ranges)
df['size'] = df['name'].str.extract(r'size_(\d+)')[0].astype(int)
df['iter'] = df['name'].str.extract(r'iter_(\d+)')[0].astype(int)

print(df.groupby('size')['duration_us'].describe())
```

### Extract Iteration Timings (Bash)

```bash
# Extract all iteration ranges
grep "iter_" results.csv | awk -F, '{print $1, $3}' | sort

# Get statistics per size
grep "iter_" results.csv | \
  awk -F, '{print $1}' | \
  sed 's/.*size_\([0-9]*\).*/\1/' | \
  sort | uniq -c
```

## Build Options

### Standard Build (no ROCTX)
```bash
make MPI=1
```

### With ROCTX
```bash
make MPI=1 USE_ROCPROFILER=1
```

### Clean and Rebuild
```bash
make MPI=1 USE_ROCPROFILER=1 clean
make MPI=1 USE_ROCPROFILER=1
```

## Verification

### Check if ROCTX is Enabled
```bash
# Should show roctxRangePush/Pop symbols
strings ./build/reduce_scatter_perf | grep roctx
```

### Check ROCProfiler Version
```bash
rocprof --version
# Requires ROCm 4.0+
```

## Performance Impact

- **CPU overhead**: ~10-50 ns per range marker
- **Memory overhead**: ~100 bytes per active range
- **GPU overhead**: None (CPU-side only)
- **Trace file size**: ~1-10 MB per 1000 iterations

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No ROCTX ranges in trace | Add `--roctx-trace` flag |
| Build fails with roctx.h not found | Check `ROCM_PATH` environment variable |
| Trace file too large | Reduce iterations with `-n 10` |
| Can't distinguish ranks | Use separate output dirs per rank |

## Integration with Existing Scripts

### Modify `run_timing_sweep.py`

Add profiling mode:

```python
def run_with_profiling(benchmark, nranks, min_size, max_size, output_dir):
    cmd = [
        "rocprof",
        "--roctx-trace",
        "--hip-trace",
        "--stats",
        f"--output-file={output_dir}/profile.csv",
        "--timestamp", "on",
        "mpirun", "-np", str(nranks),
        f"./build/{benchmark}_perf",
        "-b", str(min_size),
        "-e", str(max_size),
        "-f", "2"
    ]
    subprocess.run(cmd, check=True)
```

## See Also

- [Full ROCTX Documentation](ROCTX_USAGE.md)
- [ROCProfiler Integration Analysis](ROCPROFILER_INTEGRATION_ANALYSIS.md)
- [Wall vs Kernel Time Analysis](WALL_VS_KERNEL_TIME_ANALYSIS.md)

