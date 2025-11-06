# ROCProfiler Integration for Kernel Timing Analysis

## Problem Statement

**Goal:** Collect per-kernel timing data using ROCProfiler as an alternative to HIP events, with accurate correlation to benchmark wall-clock measurements.

**Challenge:** Correlate individual kernel launches with specific benchmark output lines (size/operation combinations) when:
- Multiple kernels may be launched per collective operation
- Warmup iterations must be distinguished from timed iterations
- MPI ranks run concurrently
- Timing data is collected externally (ROCProfiler) rather than instrumented in code

---

## Current HIP Events Approach

### How It Works

**Instrumentation Location:** `common.cu` lines 688-926

```cpp
// Inside BenchTime() - called once per size/operation
if (save_individual_timings) {
    hipHostMalloc(&timing_results, sizeof(double) * iters * agg_iters, ...);
    
    for (int iter = 0; iter < iters; iter++) {
        // Record start event
        hipEventCreate(&timing_data->start_event);
        hipEventRecord(timing_data->start_event, stream);
        
        // Launch collective
        startColl(args, type, op, root, in_place, iter);
        completeColl(args);
        
        // Record end event
        hipEventRecord(timing_data->end_event, stream);
    }
    
    // Write to CSV: size, iteration, time
    fprintf(timing_file, "%d,%.9f,%lu,%lu,...\n",
            iter, timing_results[iter], args->nbytes, ...);
}
```

**Correlation Method:**
- Each `BenchTime()` call knows `args->nbytes` (message size)
- Iteration number is tracked explicitly
- In-place vs out-of-place is a parameter
- All timing data for one size is written together

**Advantages:**
- Perfect correlation (code knows exactly what it's timing)
- Minimal overhead (HIP events are lightweight)
- No external tools required
- Works with MPI (each rank writes its own file)

**Disadvantages:**
- Requires code instrumentation
- Limited to timing (no other GPU metrics)
- HIP events may have measurement artifacts
- Adds complexity to benchmark code

---

## ROCProfiler Approaches

### Approach 1: Timestamp-Based Correlation

**Concept:** Use fine-grained timestamps to match kernel launches with benchmark phases.

#### Implementation Strategy

**Step 1: Benchmark emits timestamps**

Modify `common.cu` to write timestamp markers:

```cpp
// In BenchTime(), before timing loop
if (rocprof_correlation_enabled) {
    uint64_t start_ts = get_nanosecond_timestamp();
    write_correlation_marker(timing_file, "BEGIN_SIZE", args->nbytes, 
                             in_place, start_ts);
}

for (int iter = 0; iter < iters; iter++) {
    uint64_t iter_start = get_nanosecond_timestamp();
    
    // Launch collective
    startColl(...);
    completeColl(...);
    
    uint64_t iter_end = get_nanosecond_timestamp();
    write_correlation_marker(timing_file, "ITERATION", iter, 
                             iter_start, iter_end);
}

uint64_t end_ts = get_nanosecond_timestamp();
write_correlation_marker(timing_file, "END_SIZE", args->nbytes, end_ts);
```

**Step 2: ROCProfiler collects kernel data**

Run benchmark with ROCProfiler:

```bash
rocprof --timestamp on --stats \
        --output-file kernel_trace.csv \
        mpirun -np 8 ./reduce_scatter_perf -b 128 -e 1G ...
```

ROCProfiler output includes:
- Kernel name
- Start timestamp (ns)
- End timestamp (ns)
- Duration
- GPU ID
- Queue ID

**Step 3: Post-processing correlation**

Python script matches timestamps:

```python
def correlate_timestamps(benchmark_markers, rocprof_kernels):
    results = []
    
    for marker in benchmark_markers:
        if marker['event'] == 'ITERATION':
            # Find all kernels launched during this iteration
            iter_start = marker['start_ts']
            iter_end = marker['end_ts']
            
            matching_kernels = [
                k for k in rocprof_kernels
                if iter_start <= k['start_ts'] <= iter_end
            ]
            
            results.append({
                'size': marker['size'],
                'iteration': marker['iter'],
                'in_place': marker['in_place'],
                'kernels': matching_kernels,
                'total_kernel_time': sum(k['duration'] for k in matching_kernels)
            })
    
    return results
```

#### Advantages
- ✅ No sequence numbers needed
- ✅ Works with existing ROCProfiler
- ✅ Can collect additional GPU metrics (occupancy, memory bandwidth, etc.)
- ✅ Post-processing is flexible (can re-correlate without re-running)

#### Disadvantages
- ⚠️ Timestamp synchronization between CPU and GPU
- ⚠️ Clock drift over long runs
- ⚠️ Ambiguity if kernels overlap in time (async launches)
- ⚠️ ROCProfiler overhead may affect timing
- ⚠️ Requires careful timestamp precision (nanosecond level)

#### Timestamp Precision Requirements

**CPU Timestamp Source:**
```cpp
#include <time.h>

uint64_t get_nanosecond_timestamp() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}
```

**GPU Timestamp Source:**
- ROCProfiler uses GPU hardware counters
- Timestamps are in GPU clock domain
- May need calibration to match CPU timestamps

**Synchronization Strategy:**
```cpp
// At benchmark start, calibrate clocks
void calibrate_timestamps() {
    for (int i = 0; i < 10; i++) {
        uint64_t cpu_before = get_nanosecond_timestamp();
        hipDeviceSynchronize();  // Force GPU sync
        uint64_t cpu_after = get_nanosecond_timestamp();
        
        // Record calibration point
        // GPU timestamp at sync = (cpu_before + cpu_after) / 2
    }
}
```

---

### Approach 2: Sequence Number Injection

**Concept:** Assign a unique sequence number to each benchmark iteration and inject it into kernel launches for ROCProfiler to capture.

#### Implementation Strategy

**Step 1: Add sequence number to kernel arguments**

Modify RCCL collective wrappers to accept a correlation ID:

```cpp
// In common.cu
static uint64_t global_sequence_number = 0;

testResult_t startColl(struct threadArgs* args, ncclDataType_t type, 
                       ncclRedOp_t op, int root, int in_place, int iter) {
    uint64_t seq_num = global_sequence_number++;
    
    // Store sequence number in a GPU-accessible location
    CUDACHECK(cudaMemcpyToSymbol(d_correlation_id, &seq_num, 
                                 sizeof(uint64_t)));
    
    // Launch collective (RCCL kernels will see d_correlation_id)
    NCCLCHECK(ncclReduceScatter(...));
    
    // Record sequence number for this iteration
    write_sequence_marker(timing_file, seq_num, args->nbytes, 
                         in_place, iter);
    
    return testSuccess;
}
```

**Step 2: RCCL kernel modification (DIFFICULT)**

This requires modifying RCCL source code:

```cpp
// In RCCL kernel code (e.g., reduce_scatter.cu)
__device__ uint64_t d_correlation_id;

__global__ void ncclReduceScatterKernel(...) {
    // First thread records correlation ID
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // This would need to be captured by ROCProfiler somehow
        // Option: Write to global memory that ROCProfiler can read
        g_kernel_correlation_ids[kernel_launch_index] = d_correlation_id;
    }
    
    // ... rest of kernel ...
}
```

**Step 3: ROCProfiler with custom markers**

Use ROCProfiler API to inject markers:

```cpp
#include <rocprofiler-sdk/rocprofiler.h>

// Before kernel launch
rocprofiler_push_range("seq_%lu", seq_num);

// Launch collective
NCCLCHECK(ncclReduceScatter(...));

// After kernel launch
rocprofiler_pop_range();
```

#### Advantages
- ✅ Explicit correlation (no timestamp ambiguity)
- ✅ Works even with overlapping kernels
- ✅ No clock synchronization issues
- ✅ Robust to timing variations

#### Disadvantages
- ❌ Requires RCCL source code modification (very invasive)
- ❌ Complex integration with ROCProfiler SDK
- ❌ May not work with pre-built RCCL libraries
- ❌ Difficult to maintain across RCCL versions
- ❌ Performance overhead of marker injection

---

### Approach 3: ROCProfiler Range Markers (RECOMMENDED)

**Concept:** Use ROCProfiler's range API to mark benchmark phases without modifying RCCL.

#### Implementation Strategy

**Step 1: Instrument benchmark code with ranges**

```cpp
#include <rocprofiler-sdk/rocprofiler.h>

// In BenchTime()
char range_name[256];
snprintf(range_name, sizeof(range_name), 
         "size_%lu_%s", args->nbytes, in_place ? "inplace" : "oop");

rocprofiler_push_range(range_name);

for (int iter = 0; iter < iters; iter++) {
    char iter_name[128];
    snprintf(iter_name, sizeof(iter_name), "iter_%d", iter);
    rocprofiler_push_range(iter_name);
    
    // Launch collective
    startColl(args, type, op, root, in_place, iter);
    completeColl(args);
    
    rocprofiler_pop_range();  // iter_X
}

rocprofiler_pop_range();  // size_X_inplace/oop
```

**Step 2: Run with ROCProfiler**

```bash
rocprof --roctx-trace \
        --output-file kernel_trace.csv \
        mpirun -np 8 ./reduce_scatter_perf ...
```

**Step 3: Post-processing**

ROCProfiler output includes range information:

```
Range,KernelName,Start,End,Duration
size_128_oop/iter_0,ncclKernel_ReduceScatter_RING_LL,1000,1050,50
size_128_oop/iter_1,ncclKernel_ReduceScatter_RING_LL,1100,1145,45
...
```

Python correlation:

```python
def parse_rocprof_ranges(trace_file):
    results = defaultdict(list)
    
    for line in trace_file:
        range_path = line['Range']  # e.g., "size_128_oop/iter_5"
        kernel_name = line['KernelName']
        duration = line['Duration']
        
        # Parse range path
        parts = range_path.split('/')
        size_range = parts[0]  # "size_128_oop"
        iter_range = parts[1]  # "iter_5"
        
        size_bytes = int(size_range.split('_')[1])
        in_place = 'inplace' in size_range
        iteration = int(iter_range.split('_')[1])
        
        results[(size_bytes, in_place, iteration)].append({
            'kernel': kernel_name,
            'duration': duration
        })
    
    return results
```

#### Advantages
- ✅ No RCCL modification required
- ✅ Explicit correlation via hierarchical ranges
- ✅ Standard ROCProfiler feature (well-supported)
- ✅ Can collect full GPU metrics
- ✅ Works with MPI (each rank has separate trace)
- ✅ Minimal code changes to benchmark

#### Disadvantages
- ⚠️ Requires linking with rocprofiler-sdk
- ⚠️ Range push/pop adds small overhead
- ⚠️ Need to handle MPI rank separation in traces

---

### Approach 4: Kernel Name Pattern Matching

**Concept:** Use kernel naming conventions and launch order to infer correlation.

#### Implementation Strategy

**Step 1: Understand RCCL kernel naming**

RCCL kernels have predictable names:
```
ncclKernel_ReduceScatter_RING_LL
ncclKernel_ReduceScatter_RING_LL128
ncclKernel_ReduceScatter_TREE_LL
ncclKernel_AllReduce_RING_SIMPLE
```

**Step 2: Record expected kernel launch pattern**

```cpp
// In BenchTime()
write_expected_pattern(timing_file, args->nbytes, in_place, iters);
// Output: "Expecting 50 launches of ncclKernel_ReduceScatter_* for size 128"
```

**Step 3: Match ROCProfiler output to pattern**

```python
def correlate_by_pattern(expected_patterns, rocprof_kernels):
    kernel_iter = iter(rocprof_kernels)
    results = []
    
    for pattern in expected_patterns:
        size = pattern['size']
        expected_count = pattern['iterations']
        kernel_pattern = pattern['kernel_name_pattern']
        
        # Consume next N kernels matching pattern
        matching_kernels = []
        for _ in range(expected_count):
            kernel = next(kernel_iter)
            if re.match(kernel_pattern, kernel['name']):
                matching_kernels.append(kernel)
            else:
                # Pattern mismatch - correlation failed
                raise CorrelationError(...)
        
        results.append({
            'size': size,
            'kernels': matching_kernels
        })
    
    return results
```

#### Advantages
- ✅ No code modification required
- ✅ Works with existing ROCProfiler
- ✅ Simple implementation

#### Disadvantages
- ❌ Fragile (breaks if launch order changes)
- ❌ Cannot distinguish warmup from timed iterations
- ❌ Fails if unexpected kernels are launched
- ❌ No way to verify correlation correctness
- ❌ Doesn't work with overlapping launches

---

## Comparison Matrix

| Approach | Correlation Accuracy | Code Changes | RCCL Modification | Robustness | Overhead | Complexity |
|----------|---------------------|--------------|-------------------|------------|----------|------------|
| **HIP Events (current)** | ⭐⭐⭐⭐⭐ Perfect | Moderate | None | ⭐⭐⭐⭐⭐ | Low | Medium |
| **Timestamp-Based** | ⭐⭐⭐ Good | Moderate | None | ⭐⭐⭐ | Medium | High |
| **Sequence Number** | ⭐⭐⭐⭐⭐ Perfect | High | **Required** | ⭐⭐⭐⭐ | Medium | Very High |
| **Range Markers** | ⭐⭐⭐⭐⭐ Perfect | Moderate | None | ⭐⭐⭐⭐⭐ | Low-Medium | Medium |
| **Pattern Matching** | ⭐⭐ Poor | Minimal | None | ⭐ | Low | Low |

---

## Recommended Solution: ROCProfiler Range Markers (Approach 3)

### Why This is Best

1. **Perfect Correlation:** Hierarchical ranges explicitly mark each size/iteration
2. **No RCCL Changes:** Works with any RCCL build
3. **Standard Tool:** Uses well-supported ROCProfiler features
4. **Flexible:** Can add/remove ranges without affecting RCCL
5. **MPI Compatible:** Each rank generates separate trace
6. **Rich Data:** Can collect all GPU metrics ROCProfiler supports

### Implementation Plan

#### Phase 1: Basic Range Integration

**File:** `common.cu`

Add ROCProfiler headers and initialization:

```cpp
#ifdef USE_ROCPROFILER
#include <rocprofiler-sdk/rocprofiler.h>

static bool rocprof_initialized = false;

void init_rocprofiler() {
    if (!rocprof_initialized) {
        // Initialize ROCProfiler SDK
        rocprofiler_initialize();
        rocprof_initialized = true;
    }
}
#endif
```

Modify `BenchTime()` to add ranges:

```cpp
testResult_t BenchTime(struct threadArgs* args, ncclDataType_t type, 
                       ncclRedOp_t op, int root, int in_place) {
#ifdef USE_ROCPROFILER
    char size_range[256];
    snprintf(size_range, sizeof(size_range), 
             "benchmark_size_%lu_%s_type_%s_op_%s",
             args->nbytes, 
             in_place ? "inplace" : "oop",
             test_typenames[type],
             test_opnames[op]);
    rocprofiler_push_range(size_range);
#endif

    // ... existing warmup code ...

    for (int iter = 0; iter < iters; iter++) {
#ifdef USE_ROCPROFILER
        char iter_range[128];
        snprintf(iter_range, sizeof(iter_range), "iter_%04d", iter);
        rocprofiler_push_range(iter_range);
#endif

        // ... existing timing code ...
        TESTCHECK(startColl(args, type, op, root, in_place, iter));
        TESTCHECK(completeColl(args));

#ifdef USE_ROCPROFILER
        rocprofiler_pop_range();  // iter_XXXX
#endif
    }

#ifdef USE_ROCPROFILER
    rocprofiler_pop_range();  // benchmark_size_...
#endif

    return testSuccess;
}
```

#### Phase 2: Build System Integration

**File:** `Makefile` or `CMakeLists.txt`

```makefile
# Add ROCProfiler support
ROCPROF_ENABLED ?= 0

ifeq ($(ROCPROF_ENABLED), 1)
    CXXFLAGS += -DUSE_ROCPROFILER
    LDFLAGS += -lrocprofiler64
    INCLUDES += -I/opt/rocm/include
endif
```

Build with ROCProfiler:
```bash
make ROCPROF_ENABLED=1
```

#### Phase 3: Run Script Integration

**File:** `run_timing_sweep.py`

Add `--use-rocprofiler` option:

```python
def run_benchmark_with_rocprofiler(output_dir, benchmark_name, ...):
    """Run benchmark with ROCProfiler tracing enabled"""
    
    # Build rocprof command
    rocprof_output = os.path.join(output_dir, f"{benchmark_name}_rocprof.csv")
    
    rocprof_cmd = [
        "rocprof",
        "--roctx-trace",           # Enable range tracing
        "--hip-trace",             # Capture HIP API calls
        "--hsa-trace",             # Capture HSA operations
        "--stats",                 # Generate statistics
        f"--output-file={rocprof_output}",
        "--timestamp", "on",       # Include timestamps
    ]
    
    # Add MPI launcher
    if mpi_enabled:
        # Each rank gets separate output
        rocprof_cmd.extend([
            "--output-format", "csv",
            "--output-directory", output_dir,
        ])
        
        mpi_cmd = [
            "mpirun",
            "--bind-to", "numa",
            "-np", str(num_ranks),
        ]
        
        full_cmd = mpi_cmd + rocprof_cmd + benchmark_cmd
    else:
        full_cmd = rocprof_cmd + benchmark_cmd
    
    # Run with ROCProfiler
    result = subprocess.run(full_cmd, ...)
    
    return result
```

#### Phase 4: Analysis Script

**File:** `analyze_rocprof_timings.py`

```python
#!/usr/bin/env python3
"""
Analyze ROCProfiler trace data and correlate with benchmark results
"""

import pandas as pd
import re
from collections import defaultdict

def parse_rocprof_trace(trace_file):
    """Parse ROCProfiler CSV output with range information"""
    df = pd.read_csv(trace_file)
    
    # Filter for kernel dispatches within ranges
    kernel_rows = df[df['Name'].str.contains('ncclKernel', na=False)]
    
    results = []
    for _, row in kernel_rows.iterrows():
        # Parse range path: "benchmark_size_128_oop_type_float_op_sum/iter_0042"
        range_path = row.get('Range', '')
        if not range_path:
            continue
        
        parts = range_path.split('/')
        if len(parts) < 2:
            continue
        
        # Parse size range
        size_match = re.search(r'size_(\d+)', parts[0])
        inplace_match = re.search(r'_(inplace|oop)', parts[0])
        type_match = re.search(r'type_(\w+)', parts[0])
        op_match = re.search(r'op_(\w+)', parts[0])
        
        # Parse iteration
        iter_match = re.search(r'iter_(\d+)', parts[1])
        
        if all([size_match, inplace_match, iter_match]):
            results.append({
                'size_bytes': int(size_match.group(1)),
                'in_place': inplace_match.group(1) == 'inplace',
                'data_type': type_match.group(1) if type_match else 'unknown',
                'operation': op_match.group(1) if op_match else 'unknown',
                'iteration': int(iter_match.group(1)),
                'kernel_name': row['Name'],
                'duration_ns': row['DurationNs'],
                'start_ns': row['BeginNs'],
                'end_ns': row['EndNs'],
                'gpu_id': row.get('DeviceId', -1),
                'queue_id': row.get('QueueId', -1),
            })
    
    return pd.DataFrame(results)

def aggregate_by_size(rocprof_df):
    """Aggregate kernel timings by size/operation"""
    
    grouped = rocprof_df.groupby(['size_bytes', 'in_place', 'data_type', 'operation'])
    
    stats = grouped['duration_ns'].agg([
        ('count', 'count'),
        ('mean_ns', 'mean'),
        ('std_ns', 'std'),
        ('min_ns', 'min'),
        ('max_ns', 'max'),
        ('p50_ns', lambda x: x.quantile(0.50)),
        ('p95_ns', lambda x: x.quantile(0.95)),
        ('p99_ns', lambda x: x.quantile(0.99)),
    ]).reset_index()
    
    # Convert to microseconds
    for col in ['mean_ns', 'std_ns', 'min_ns', 'max_ns', 'p50_ns', 'p95_ns', 'p99_ns']:
        stats[col.replace('_ns', '_us')] = stats[col] / 1000.0
    
    return stats

def correlate_with_benchmark(rocprof_df, benchmark_output_file):
    """Correlate ROCProfiler data with benchmark wall-clock times"""
    
    # Load benchmark output
    bench_df = parse_benchmark_output(benchmark_output_file)
    
    # Aggregate ROCProfiler data
    rocprof_stats = aggregate_by_size(rocprof_df)
    
    # Merge on size/operation
    merged = pd.merge(
        rocprof_stats,
        bench_df,
        on=['size_bytes', 'in_place', 'data_type', 'operation'],
        how='outer',
        suffixes=('_rocprof', '_benchmark')
    )
    
    # Calculate divergence
    merged['kernel_vs_wall_ratio'] = merged['mean_us'] / merged['wall_time_us']
    
    return merged

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze ROCProfiler timing data')
    parser.add_argument('--rocprof-trace', required=True, 
                       help='ROCProfiler CSV trace file')
    parser.add_argument('--benchmark-output', required=True,
                       help='Benchmark output text file')
    parser.add_argument('--output', default='rocprof_analysis.csv',
                       help='Output CSV file')
    
    args = parser.parse_args()
    
    # Parse ROCProfiler trace
    print(f"Parsing ROCProfiler trace: {args.rocprof_trace}")
    rocprof_df = parse_rocprof_trace(args.rocprof_trace)
    print(f"Found {len(rocprof_df)} kernel launches")
    
    # Correlate with benchmark
    print(f"Correlating with benchmark output: {args.benchmark_output}")
    merged_df = correlate_with_benchmark(rocprof_df, args.benchmark_output)
    
    # Save results
    merged_df.to_csv(args.output, index=False)
    print(f"Analysis saved to: {args.output}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(merged_df[['size_bytes', 'mean_us', 'wall_time_us', 'kernel_vs_wall_ratio']])

if __name__ == '__main__':
    main()
```

---

## Alternative: Hybrid Approach

**Concept:** Use HIP events for basic timing, ROCProfiler for detailed analysis.

### When to Use Each

**HIP Events (default):**
- Regular performance testing
- CI/CD pipelines
- Quick turnaround
- Minimal overhead

**ROCProfiler (detailed analysis):**
- Performance debugging
- Algorithm analysis
- Understanding kernel behavior
- Collecting GPU metrics (occupancy, memory bandwidth, etc.)

### Implementation

Add `--profiling-mode` option to `run_timing_sweep.py`:

```python
parser.add_argument('--profiling-mode', 
                   choices=['hip-events', 'rocprofiler', 'both'],
                   default='hip-events',
                   help='Timing collection method')
```

Build with conditional compilation:

```cpp
#if defined(USE_HIP_EVENTS)
    // Current HIP events code
#elif defined(USE_ROCPROFILER)
    // ROCProfiler range markers
#endif
```

---

## MPI Considerations

### Per-Rank Trace Files

ROCProfiler with MPI generates separate trace files per rank:

```
output_dir/
├── reduce_scatter_rocprof_rank0.csv
├── reduce_scatter_rocprof_rank1.csv
├── ...
└── reduce_scatter_rocprof_rank7.csv
```

### Aggregation Strategy

```python
def aggregate_mpi_traces(output_dir, benchmark_name, num_ranks):
    """Combine traces from all MPI ranks"""
    
    all_traces = []
    for rank in range(num_ranks):
        trace_file = f"{output_dir}/{benchmark_name}_rocprof_rank{rank}.csv"
        df = parse_rocprof_trace(trace_file)
        df['mpi_rank'] = rank
        all_traces.append(df)
    
    combined = pd.concat(all_traces, ignore_index=True)
    
    # Aggregate across ranks
    stats = combined.groupby(['size_bytes', 'in_place', 'iteration']).agg({
        'duration_ns': ['mean', 'std', 'min', 'max'],
        'mpi_rank': 'count'  # Verify all ranks present
    })
    
    return stats
```

---

## Performance Overhead Analysis

### HIP Events Overhead

**Measurement:** ~1-5 µs per event pair
**Impact:** Negligible for kernels > 100 µs
**Scaling:** O(iterations)

### ROCProfiler Overhead

**Measurement:** ~10-50 µs per range push/pop
**Impact:** 
- Small for large messages (> 1 MiB)
- Noticeable for small messages (< 1 KiB)
**Scaling:** O(iterations)

**Mitigation:**
- Use ROCProfiler only for detailed analysis, not routine testing
- Reduce iteration count when profiling
- Use sampling mode if available

---

## Recommendations

### Primary Recommendation: ROCProfiler Range Markers

**Implement Approach 3** for the following reasons:

1. **No RCCL modification** - works with any build
2. **Explicit correlation** - no ambiguity
3. **Standard tool** - well-supported by AMD
4. **Extensible** - can add more metrics easily
5. **MPI compatible** - works with multi-rank runs

### Implementation Priority

**Phase 1 (High Priority):**
- Add ROCProfiler range markers to `BenchTime()`
- Create build flag `USE_ROCPROFILER`
- Test with single-rank runs

**Phase 2 (Medium Priority):**
- Integrate with `run_timing_sweep.py`
- Create `analyze_rocprof_timings.py`
- Test with MPI runs

**Phase 3 (Low Priority):**
- Add hybrid mode (HIP events + ROCProfiler)
- Collect additional GPU metrics
- Create visualization tools

### Fallback: Timestamp-Based Correlation

If ROCProfiler range markers prove problematic:
- Implement **Approach 1** (timestamp-based)
- Requires careful clock calibration
- More complex post-processing
- But still doesn't require RCCL changes

---

## Example Usage

### With HIP Events (current)

```bash
python run_timing_sweep.py reduce_scatter \
    --mpi --ranks 8 \
    --iterations 50
```

### With ROCProfiler

```bash
# Build with ROCProfiler support
cd rccl-tests
make ROCPROF_ENABLED=1

# Run with profiling
python run_timing_sweep.py reduce_scatter \
    --mpi --ranks 8 \
    --iterations 50 \
    --profiling-mode rocprofiler

# Analyze results
python analyze_rocprof_timings.py \
    --rocprof-trace data/hostname/run_reduce_scatter_*/reduce_scatter_rocprof_rank0.csv \
    --benchmark-output data/hostname/run_reduce_scatter_*/reduce_scatter_benchmark_output.txt \
    --output reduce_scatter_rocprof_analysis.csv
```

---

## Conclusion

**ROCProfiler Range Markers (Approach 3)** provides the best balance of:
- Correlation accuracy
- Implementation complexity
- Maintainability
- Feature richness

This approach allows collecting detailed GPU metrics without modifying RCCL, while maintaining perfect correlation between kernel launches and benchmark measurements.

The implementation can be done incrementally, starting with basic range markers and expanding to full metric collection over time.

---

## References

1. ROCProfiler Documentation: https://rocm.docs.amd.com/projects/rocprofiler/en/latest/
2. ROCProfiler SDK: https://github.com/ROCm/rocprofiler-sdk
3. ROCTX API: https://rocm.docs.amd.com/projects/roctracer/en/latest/
4. HIP Events: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html#events

---

## See Also

- `WALL_VS_KERNEL_TIME_ANALYSIS.md` - Understanding timing divergence
- `ANALYZE_TIMING_STATS.md` - Current HIP events analysis
- `SCRIPT_ECOSYSTEM.md` - Overall script documentation

