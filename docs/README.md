# RCCL Performance Analysis Ecosystem

**Complete guide to the RCCL performance analysis and benchmarking system**

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Production Workflow](#production-workflow)
4. [System Architecture](#system-architecture)
5. [Script Reference](#script-reference)
6. [Data Organization](#data-organization)
7. [Analysis Capabilities](#analysis-capabilities)
8. [Technical Details](#technical-details)
9. [Troubleshooting](#troubleshooting)
10. [Project Accomplishments](#project-accomplishments)

---

## Overview

This project provides a comprehensive RCCL (ROCm Communication Library) performance analysis and benchmarking ecosystem. The system enables systematic, quantitative analysis of collective communication performance across different configurations, scales, and hardware architectures.

**Key Features:**
- Automated benchmark execution with ROCProfiler integration
- Per-kernel GPU timing with microsecond precision
- Statistical analysis with BIC-based performance segmentation
- Interactive Plotly visualizations
- Multi-rank MPI support
- Comprehensive documentation

**Status:** ✅ Production-ready and operational

---

## Quick Start

### Prerequisites

```bash
# Environment setup (already configured)
export LD_LIBRARY_PATH=/work/lmeadows/rccl/install/lib:/opt/openmpi-5.0.8-Rel7.0.0/lib:$LD_LIBRARY_PATH
export PATH=/opt/openmpi-5.0.8-Rel7.0.0/bin:$PATH
```

### Run a Complete Analysis (5 minutes)

```bash
cd /work/lmeadows/rccl/rccl-tests

# 1. Run benchmark with 8 ranks (collects ROCProfiler data)
python3 scripts/run_timing_sweep.py all_reduce --ranks 8

# 2. Analyze the results (use the most recent run directory)
RUN_DIR=$(ls -td /work/lmeadows/rccl/data/$(hostname)/run_all_reduce_* | head -1)

# 3. Run statistical analysis
python3 scripts/analyze_timing_stats.py "$RUN_DIR"

# 4. Perform BIC segmentation
python3 scripts/segment_performance_bic.py "$RUN_DIR"

# 5. Create visualizations
python3 scripts/plot_size_vs_time_plotly.py "$RUN_DIR"
python3 scripts/create_boxplots.py "$RUN_DIR"
python3 scripts/plot_kernel_timeline.py "$RUN_DIR"

# 6. View results
echo "Results in: $RUN_DIR"
ls -lh "$RUN_DIR"/*.html "$RUN_DIR"/*.png
```

---

## Production Workflow

### Complete End-to-End Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    1. DATA COLLECTION                           │
│                  run_timing_sweep.py                            │
│                                                                 │
│  • Runs benchmark with ROCProfiler kernel tracing              │
│  • Collects wall-clock times from benchmark output            │
│  • Captures per-rank timestamps (CLOCK_BOOTTIME)               │
│  • Saves ROCProfiler kernel traces (CSV format)                │
│                                                                 │
│  Output: /work/lmeadows/rccl/data/<hostname>/run_<bench>_*/   │
│    ├── <benchmark>_benchmark_output.txt                        │
│    ├── rank_N_timestamps.txt                                   │
│    ├── rank_pid_mapping.json                                   │
│    └── rocp/<hostname>/<pid>_kernel_trace.csv                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    2. TIMING CORRELATION                        │
│              correlate_rocprof_timings.py                       │
│                (automatically called by step 1)                 │
│                                                                 │
│  • Parses benchmark timestamps (Tstart/Tend markers)           │
│  • Parses ROCProfiler kernel traces                            │
│  • Correlates kernels with benchmark runs (strict containment) │
│  • Generates per-rank timing CSVs                              │
│                                                                 │
│  Output: all_rank*.csv (size, inplace, iteration, time_sec)    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   3. STATISTICAL ANALYSIS                       │
│               analyze_timing_stats.py                           │
│                                                                 │
│  • Loads all_rank*.csv files                                   │
│  • Computes statistics (mean, std, percentiles)                │
│  • Aggregates across ranks                                     │
│  • Generates summary reports                                   │
│                                                                 │
│  Output: <benchmark>_timing_analysis.txt                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   4. PERFORMANCE SEGMENTATION                   │
│              segment_performance_bic.py                         │
│                                                                 │
│  • Applies Piecewise Linear Regression with BIC                │
│  • Identifies exactly 3 performance segments                   │
│  • Determines log-linear vs linear model per segment           │
│  • Calculates breakpoints and R² values                        │
│                                                                 │
│  Output: <benchmark>_bic_segmentation.json                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      5. VISUALIZATION                           │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  plot_size_vs_time_plotly.py                              │ │
│  │  • Interactive size vs. time plot                         │ │
│  │  • Separate OOP/INP traces                                │ │
│  │  • IQR error bands (25-75 percentile)                     │ │
│  │  • BIC segment boundaries                                 │ │
│  │  • Wall clock times overlay                               │ │
│  │  Output: <benchmark>_size_vs_time.html                    │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  create_boxplots.py                                       │ │
│  │  • One figure per BIC segment                             │ │
│  │  • Boxplots for all sizes in segment                      │ │
│  │  • OOP/INP side-by-side comparison                        │ │
│  │  • Adaptive Y-axis (linear/log per segment)               │ │
│  │  Output: <benchmark>_segment*_boxplots.png                │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  plot_kernel_timeline.py                                  │ │
│  │  • Interactive kernel execution timeline                  │ │
│  │  • One row per MPI rank                                   │ │
│  │  • Tstart/Tend markers                                    │ │
│  │  • Zoomable/pannable interface                            │ │
│  │  Output: kernel_timeline.html                             │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Instructions

#### Step 1: Run Benchmark

```bash
cd /work/lmeadows/rccl/rccl-tests

# Basic usage (8 ranks, 8B to 1GiB sweep)
python3 scripts/run_timing_sweep.py <benchmark_name> --ranks 8

# Custom configuration
python3 scripts/run_timing_sweep.py all_reduce \
    --ranks 8 \
    --min-size 128 \
    --max-size 512M \
    --iterations 100 \
    --warmup 5

# Available benchmarks:
# all_reduce, all_gather, broadcast, reduce_scatter, scatter, gather,
# alltoall, alltoallv, hypercube, sendrecv, all_reduce_bias
```

**What happens:**
- Benchmark runs with `rocprofv3` kernel tracing
- Creates directory: `/work/lmeadows/rccl/data/<hostname>/run_<benchmark>_YYYYMMDD_HHMMSS/`
- Collects benchmark output, timestamps, and kernel traces
- Automatically correlates timing data
- Generates `all_rank*.csv` files

#### Step 2: Analyze Statistics

```bash
# Find the most recent run directory
RUN_DIR=$(ls -td /work/lmeadows/rccl/data/$(hostname)/run_<benchmark>_* | head -1)

# Run statistical analysis
python3 scripts/analyze_timing_stats.py "$RUN_DIR"
```

**Output:**
- Console: Statistical summary (mean, std, percentiles)
- File: `<benchmark>_timing_analysis.txt`

#### Step 3: Perform Segmentation

```bash
python3 scripts/segment_performance_bic.py "$RUN_DIR"
```

**Output:**
- Console: Segment boundaries and model fits
- File: `<benchmark>_bic_segmentation.json`

#### Step 4: Create Visualizations

```bash
# Interactive size vs. time plot
python3 scripts/plot_size_vs_time_plotly.py "$RUN_DIR"

# Boxplots per segment
python3 scripts/create_boxplots.py "$RUN_DIR"

# Kernel timeline
python3 scripts/plot_kernel_timeline.py "$RUN_DIR"
```

**Output:**
- `<benchmark>_size_vs_time.html` - Interactive Plotly chart
- `<benchmark>_segment*_boxplots.png` - PNG boxplots per segment
- `kernel_timeline.html` - Interactive timeline

#### Step 5: View Results

```bash
# List all generated files
ls -lh "$RUN_DIR"/*.html "$RUN_DIR"/*.png "$RUN_DIR"/*.json "$RUN_DIR"/*.txt

# Open in browser (example)
firefox "$RUN_DIR/<benchmark>_size_vs_time.html"
```

---

## System Architecture

### Data Flow Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Benchmark   │────▶│  ROCProfiler │────▶│  Correlation │────▶│   Analysis   │
│  Execution   │     │  Collection  │     │   & Parsing  │     │ & Visuals    │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
       │                     │                     │                     │
       ▼                     ▼                     ▼                     ▼
  Wall-clock           Kernel traces        all_rank*.csv         HTML/PNG
  timestamps           PID mapping          Correlated data       Reports
```

### Key Components

#### 1. Benchmark Instrumentation (`src/common.cu`)
- **Timestamp generation**: `CLOCK_BOOTTIME` for ROCProfiler correlation
- **File output**: Per-rank timestamp files (`rank_N_timestamps.txt`)
- **Markers**: `Tstart` and `Tend` events for each benchmark run
- **Compilation**: Requires `USE_ROCPROFILER=1` flag

#### 2. ROCProfiler Integration
- **Tool**: `rocprofv3` with `--kernel-trace` flag
- **Output**: CSV files with kernel dispatch information
- **Correlation**: Strict containment matching (no tolerance window)
- **Timebase**: `CLOCK_BOOTTIME` for accurate timestamp alignment

#### 3. Data Processing
- **Correlation**: `correlate_rocprof_timings.py` matches kernels to benchmark runs
- **Analysis**: `analyze_timing_stats.py` computes statistics across ranks
- **Segmentation**: `segment_performance_bic.py` identifies performance regimes

#### 4. Visualization
- **Plotly**: Interactive HTML charts with zoom/pan
- **Matplotlib**: High-quality PNG boxplots
- **Segmentation overlay**: Visual segment boundaries

---

## Script Reference

### Core Production Scripts

#### `run_timing_sweep.py`
**Purpose:** Main data collection script with ROCProfiler integration

**Usage:**
```bash
python3 scripts/run_timing_sweep.py <benchmark> [options]

Options:
  --ranks N           Number of MPI ranks (default: 8)
  --min-size SIZE     Minimum message size (default: 8)
  --max-size SIZE     Maximum message size (default: 1G)
  --iterations N      Number of timed iterations (default: 100)
  --warmup N          Number of warmup iterations (default: 5)
```

**Features:**
- Automatic minimum size adjustment for alignment-affected benchmarks
- ROCProfiler kernel trace collection
- Rank-to-PID mapping
- Automatic correlation of timing data

**Output Location:** `/work/lmeadows/rccl/data/<hostname>/run_<benchmark>_YYYYMMDD_HHMMSS/`

---

#### `correlate_rocprof_timings.py`
**Purpose:** Correlate benchmark timestamps with ROCProfiler kernel traces

**Usage:**
```bash
python3 scripts/correlate_rocprof_timings.py <run_directory>
```

**Features:**
- Parses `rank_N_timestamps.txt` files
- Parses ROCProfiler `*_kernel_trace.csv` files
- Strict containment matching (kernels must fall within Tstart/Tend)
- Generates `all_rank*.csv` with individual kernel durations

**Note:** Automatically called by `run_timing_sweep.py`

---

#### `analyze_timing_stats.py`
**Purpose:** Statistical analysis of timing data

**Usage:**
```bash
python3 scripts/analyze_timing_stats.py <run_directory>
```

**Output:**
- Overall statistics (mean, median, std, min, max)
- Per-operation mode statistics (OOP vs INP)
- Sample size statistics
- Console and file output

**Documentation:** See `docs/ANALYZE_TIMING_STATS.md`

---

#### `segment_performance_bic.py`
**Purpose:** Piecewise Linear Regression with Bayesian Information Criterion

**Usage:**
```bash
python3 scripts/segment_performance_bic.py <run_directory>
```

**Features:**
- Exactly 3 segments per benchmark
- Automatic model selection (log-linear vs linear)
- Breakpoint identification
- R² and BIC values per segment

**Output:** `<benchmark>_bic_segmentation.json`

**Documentation:** See `docs/SEGMENT_PERFORMANCE_BIC.md`

---

#### `plot_size_vs_time_plotly.py`
**Purpose:** Interactive Plotly visualization of size vs. time

**Usage:**
```bash
python3 scripts/plot_size_vs_time_plotly.py <run_directory>
```

**Features:**
- Separate traces for OOP and INP
- Kernel mean with IQR error bands (25-75 percentile)
- Wall clock times overlay
- BIC segment boundaries as vertical lines
- Log-log scale
- Interactive zoom/pan/hover

**Output:** `<benchmark>_size_vs_time.html`

---

#### `create_boxplots.py`
**Purpose:** Boxplot visualizations per BIC segment

**Usage:**
```bash
python3 scripts/create_boxplots.py <run_directory>
```

**Features:**
- One figure per BIC segment
- All sizes in segment shown side-by-side
- OOP/INP comparison
- Adaptive Y-axis (linear/log based on segment model)
- Mean values displayed
- Automatic 2-row layout for segments with >7 sizes

**Output:** `<benchmark>_segment*_boxplots.png`

---

#### `plot_kernel_timeline.py`
**Purpose:** Interactive kernel execution timeline

**Usage:**
```bash
python3 scripts/plot_kernel_timeline.py <run_directory>
```

**Features:**
- One row per MPI rank
- Horizontal bars for kernel execution
- Tstart/Tend markers as vertical lines
- Interactive hover information
- Zoomable/pannable interface

**Output:** `kernel_timeline.html`

---

### Specialized Scripts

#### `analyze_nccl_kernel_counts.py`
**Purpose:** Analyze NCCL kernel counts and durations

**Usage:**
```bash
python3 scripts/analyze_nccl_kernel_counts.py <run_directory>
```

**Output:** Tables showing kernel counts and durations per size/rank

---

#### `compare_rccL_gpu.py`
**Purpose:** Compare RCCL performance vs. raw GPU interconnect

**Usage:**
```bash
python3 scripts/compare_rccL_gpu.py <rccl_output> <gpu_output>
```

---

#### `visualize_gpu_benchmark.py`
**Purpose:** Visualize GPU transfer benchmark data

**Usage:**
```bash
python3 scripts/visualize_gpu_benchmark.py <gpu_benchmark_output>
```

---

## Data Organization

### Directory Structure

```
/work/lmeadows/rccl/
├── rccl-tests/                      # Main repository
│   ├── src/                         # Source code
│   │   └── common.cu                # Instrumented with timestamps
│   ├── build/                       # Compiled benchmarks
│   │   └── *_perf                   # Benchmark executables
│   ├── scripts/                     # Analysis scripts
│   │   ├── run_timing_sweep.py
│   │   ├── correlate_rocprof_timings.py
│   │   ├── analyze_timing_stats.py
│   │   ├── segment_performance_bic.py
│   │   ├── plot_size_vs_time_plotly.py
│   │   ├── create_boxplots.py
│   │   └── plot_kernel_timeline.py
│   └── docs/                        # Documentation
│       ├── README.md                # This file
│       ├── ANALYZE_TIMING_STATS.md
│       └── SEGMENT_PERFORMANCE_BIC.md
│
└── data/                            # Generated data (not in git)
    └── <hostname>/                  # Per-machine results
        └── run_<benchmark>_YYYYMMDD_HHMMSS/
            ├── <benchmark>_benchmark_output.txt
            ├── rank_N_timestamps.txt
            ├── rank_pid_mapping.json
            ├── all_rank*.csv
            ├── <benchmark>_timing_analysis.txt
            ├── <benchmark>_bic_segmentation.json
            ├── <benchmark>_size_vs_time.html
            ├── <benchmark>_segment*_boxplots.png
            └── kernel_timeline.html
```

### File Formats

#### `rank_N_timestamps.txt`
```
Tstart 0: 1234567890123456: all_reduce_size_1024_oop_type_float_op_sum_root_0
Tend 0: 1234567890234567
Tstart 0: 1234567890345678: all_reduce_size_1024_inp_type_float_op_sum_root_0
Tend 0: 1234567890456789
...
```

#### `all_rankN.csv`
```
size_bytes,inplace,iteration,time_seconds
8,0,0,0.000015
8,0,1,0.000014
...
```

#### `<benchmark>_bic_segmentation.json`
```json
{
  "benchmark": "all_reduce",
  "n_segments": 3,
  "bic": 88.59,
  "breakpoint_sizes": [1024, 8388608],
  "segments": [
    {
      "segment": 0,
      "size_range_bytes": [8, 512],
      "model": "log-linear",
      "r_squared": 0.9876,
      "n_points": 7
    },
    ...
  ]
}
```

---

## Analysis Capabilities

### Statistical Metrics

**Per-size statistics (aggregated across ranks):**
- Mean, median, standard deviation
- Min, max
- 25th, 50th, 75th, 95th, 99th percentiles
- Coefficient of variation (CV%)
- Sample count

**Comparison metrics:**
- Wall clock time vs. kernel time
- Out-of-place vs. in-place
- Across message sizes
- Across MPI ranks

### Performance Segmentation

**BIC-based segmentation identifies:**
1. **Latency-dominated regime** (small messages)
   - Typically log-linear relationship
   - Fixed overhead dominates
   
2. **Transition regime** (medium messages)
   - Often linear relationship
   - Mixed latency/bandwidth effects
   
3. **Bandwidth-dominated regime** (large messages)
   - Typically linear relationship
   - Transfer time dominates

### Visualization Features

**Interactive Plotly charts:**
- Zoom, pan, hover for details
- Toggle traces on/off
- Export to PNG
- Responsive layout

**Boxplots:**
- Distribution visualization
- Outlier detection
- Quartile ranges
- Mean value annotations

**Timeline:**
- Kernel execution patterns
- Rank synchronization
- Timing correlation validation

---

## Technical Details

### ROCProfiler Integration

#### Compilation
```bash
cd /work/lmeadows/rccl/rccl-tests
./domake  # Automatically includes USE_ROCPROFILER=1
```

#### Timestamp Correlation
- **Clock source**: `CLOCK_BOOTTIME` (monotonic, includes suspend time)
- **Precision**: Nanosecond
- **Correlation method**: Strict containment (kernel start/end within Tstart/Tend)
- **No tolerance window**: Ensures accurate kernel-to-benchmark matching

#### ROCProfiler Output
- **Format**: CSV with `KERNEL_DISPATCH` events
- **Fields**: `Kernel_Name`, `Start_Timestamp`, `End_Timestamp`
- **Location**: `rocp/<hostname>/<pid>_kernel_trace.csv`

### Alignment Issue Handling

Six benchmarks (`all_gather`, `gather`, `scatter`, `reduce_scatter`, `alltoall`, `hypercube`) use aggressive 16-byte alignment that can cause zero-size outputs for small messages with multiple ranks.

**Automatic fix in `run_timing_sweep.py`:**
- Detects affected benchmarks
- Calculates safe minimum: `min_size = nranks × 16` (rounded to power of 2)
- Adjusts sweep range automatically
- Reports adjustment to user

**Documentation:** See `docs/ALIGNMENT_ISSUE_ANALYSIS.md`

### Performance Considerations

**Typical benchmark run time (8 ranks, 8B-1GiB):**
- ~5-10 minutes per benchmark
- Depends on message sizes and iteration count

**Disk space per run:**
- ~50-200 MB (includes ROCProfiler traces)
- Varies with number of ranks and iterations

**Analysis time:**
- Statistics: <1 second
- Segmentation: <1 second
- Plotly visualization: 1-2 seconds
- Boxplots: 2-3 seconds
- Timeline: 2-3 seconds

---

## Troubleshooting

### Common Issues

#### 1. No timestamp files generated

**Symptom:** `rank_N_timestamps.txt` files missing after benchmark run

**Cause:** Benchmarks not compiled with `USE_ROCPROFILER=1`

**Solution:**
```bash
cd /work/lmeadows/rccl/rccl-tests
./domake  # Rebuilds with correct flags
```

---

#### 2. Correlation reports 0 kernels matched

**Symptom:** `correlate_rocprof_timings.py` finds no matching kernels

**Possible causes:**
- Clock mismatch (should use `CLOCK_BOOTTIME`)
- ROCProfiler traces missing or empty
- Timestamp files missing

**Solution:**
```bash
# Check timestamp files exist
ls -lh <run_dir>/rank_*_timestamps.txt

# Check ROCProfiler traces exist
ls -lh <run_dir>/rocp/$(hostname)/*_kernel_trace.csv

# Verify clock source in src/common.cu (line 37)
grep CLOCK_BOOTTIME src/common.cu
```

---

#### 3. Missing wall clock times for small sizes

**Symptom:** Benchmark output missing times for sizes 8, 16, 32, 64

**Cause:** Alignment issue causing zero-size outputs

**Solution:** `run_timing_sweep.py` automatically adjusts minimum size. If running benchmark manually, use appropriate minimum:
```bash
# For 8 ranks, affected benchmarks need minimum 128 bytes
./build/reduce_scatter_perf -b 128 -e 1G -f 2 -n 100
```

---

#### 4. Plotly chart not showing segment lines

**Symptom:** Vertical segment lines missing from interactive chart

**Cause:** BIC segmentation not run or JSON file missing

**Solution:**
```bash
# Run segmentation first
python3 scripts/segment_performance_bic.py <run_dir>

# Then create plot
python3 scripts/plot_size_vs_time_plotly.py <run_dir>
```

---

#### 5. Large error bars at first size point

**Symptom:** Unrealistic error bars on small message sizes

**Cause:** Outliers in first few iterations (warmup effects)

**Note:** This is expected behavior. The IQR (25-75 percentile) error bands handle outliers better than standard deviation. If needed, increase warmup iterations:
```bash
python3 scripts/run_timing_sweep.py <benchmark> --warmup 10
```

---

### Debug Mode

For detailed debugging, examine intermediate files:

```bash
RUN_DIR=<your_run_directory>

# Check benchmark output
cat "$RUN_DIR/<benchmark>_benchmark_output.txt"

# Check timestamp events
head -20 "$RUN_DIR/rank_0_timestamps.txt"

# Check correlation results
head -20 "$RUN_DIR/all_rank0.csv"

# Check ROCProfiler traces
head -20 "$RUN_DIR/rocp/$(hostname)/"*_kernel_trace.csv
```

---

## Project Accomplishments

### Major Achievements

#### ✅ 1. RCCL Test Build & Verification
- **Build System**: Functional RCCL tests compilation via `./domake`
- **MPI Integration**: Full OpenMPI 5.0.8 support with proper library configuration
- **Benchmark Suite**: Complete set of collective operation benchmarks:
  - `all_reduce_perf`, `all_gather_perf`, `broadcast_perf`
  - `reduce_scatter_perf`, `scatter_perf`, `gather_perf`
  - `alltoall_perf`, `alltoallv_perf`, `hypercube_perf`, `sendrecv_perf`
  - `all_reduce_bias_perf`
- **Environment Setup**: Complete LD_LIBRARY_PATH and PATH configuration

#### ✅ 2. ROCProfiler Integration
- **Timestamp Correlation**: `CLOCK_BOOTTIME` for accurate kernel matching
- **Per-Kernel Granularity**: Individual timing measurement for each collective operation
- **MPI Rank Support**: Proper rank-based file naming in distributed environments
- **Strict Containment**: Accurate correlation without tolerance windows
- **Data Format**: Structured CSV files with comprehensive metadata

#### ✅ 3. Automated Timing Sweep Infrastructure
- **Core Script**: `run_timing_sweep.py` - Automated benchmark execution engine
- **Size Coverage**: Comprehensive sweep from 8 bytes to 1 GiB (28 power-of-2 sizes)
- **Configuration Flexibility**: Support for different iterations, warmup, and rank counts
- **MPI Support**: Multi-rank execution with proper process isolation
- **Output Organization**: Timestamped result directories with complete datasets
- **Alignment Fix**: Automatic minimum size adjustment for affected benchmarks

#### ✅ 4. Statistical Analysis & Visualization Framework
- **Analysis Engine**: `analyze_timing_stats.py` - Comprehensive statistical analysis
- **BIC Segmentation**: `segment_performance_bic.py` - Principled performance regime identification
- **Interactive Plotly**: Size vs. time with IQR error bands and segment boundaries
- **Boxplot Visualization**: Per-segment distribution analysis with adaptive scaling
- **Kernel Timeline**: Interactive timeline showing kernel execution across ranks
- **Statistical Metrics**: Mean, std, percentiles, CV across ranks and iterations

#### ✅ 5. Complete Documentation & Organization
- **Main Guide**: This README - Complete workflow and reference
- **Script Documentation**: Individual guides for key scripts
- **Historical Documentation**: Analysis records and investigation results
- **File Organization**: Clean separation of docs, results, and source code
- **Best Practices**: Coding guidelines and quality standards

### Technical Architecture Highlights

**Data Flow Pipeline:**
```
Benchmark → ROCProfiler → Correlation → Analysis → Segmentation → Visualization
```

**Key Technical Features:**
- GPU kernel-level timing precision (microseconds)
- Statistical power: 100 iterations × 28 sizes × 2 modes = 5,600+ measurements per run
- Overhead quantification between GPU operations and application-level timing
- Scalability analysis across message sizes and operation types
- Automated performance regime identification

### Quality Standards

**Coding Guidelines:**
- Buffer safety: `snprintf` with size parameters
- Error handling: Comprehensive validation and reporting
- Documentation: Complete usage guides and API references
- Reproducibility: Timestamped experiments with full metadata

**File Organization:**
- Documentation: `/docs/` for all guides
- Results: `/data/<hostname>/` for generated data
- Scripts: `/scripts/` for analysis tools
- Source: `/src/` and `/build/` for code and binaries

### Impact & Value

**Research & Development:**
- Systematic performance analysis of RCCL collective operations
- Optimization insights through performance regime identification
- Scalability characterization across message sizes
- Hardware utilization analysis

**Engineering:**
- Automated testing for performance regression detection
- Configuration management across different parameters
- Result correlation linking GPU operations to application performance
- Statistical confidence with robust measurement techniques

**Operational:**
- Time efficiency through automated analysis pipeline
- Data integrity via structured formats
- Knowledge preservation through comprehensive documentation
- Tool reusability via modular design

---

## Future Extensions

### Potential Enhancements

**Additional Analysis:**
- More collective operations and configurations
- Hardware counters via ROCProfiler
- Cross-version and cross-hardware comparisons
- Real-time monitoring in production

**Scalability:**
- Larger scale testing (more ranks, GPUs)
- Distributed analysis across compute nodes
- Database integration for long-term trending

**Visualization:**
- 3D performance surfaces
- Comparative dashboards
- Real-time monitoring displays

---

## Appendix

### Environment Variables

```bash
# RCCL installation
export NCCL_HOME=/work/lmeadows/rccl/install

# MPI installation
export MPI_HOME=/opt/openmpi-5.0.8-Rel7.0.0

# Library paths
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$MPI_HOME/lib:$LD_LIBRARY_PATH

# Binary paths
export PATH=$MPI_HOME/bin:$PATH
```

### Build Configuration

```bash
# Build RCCL tests with MPI and ROCProfiler support
cd /work/lmeadows/rccl/rccl-tests
./domake

# Manual build (if needed)
cd src
make MPI=1 USE_ROCPROFILER=1
```

### Benchmark Parameters

**Common flags:**
- `-b <size>`: Minimum message size
- `-e <size>`: Maximum message size
- `-f <factor>`: Step factor (2 = double each step)
- `-n <count>`: Number of iterations
- `-w <count>`: Warmup iterations
- `-g <count>`: GPUs per process (usually 1)

**Size suffixes:**
- `K`: Kilobytes (1024)
- `M`: Megabytes (1024²)
- `G`: Gigabytes (1024³)

### Contact & Support

For questions or issues:
1. Check this documentation
2. Review historical docs in `/docs/`
3. Examine script docstrings
4. Check AI_GUIDELINES.md for project conventions

---

**Document Version:** 1.0  
**Last Updated:** November 6, 2025  
**Status:** ✅ Production Ready

