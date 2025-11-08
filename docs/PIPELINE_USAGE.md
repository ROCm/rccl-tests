# Complete Pipeline Script Usage

## Overview

The `run_full_pipeline.py` script orchestrates the entire RCCL performance analysis workflow, running all steps in the correct order and generating all possible outputs.

## Pipeline Steps

The script executes these steps in order:

1. **Run Benchmark** - Execute benchmark with timing sweep (`run_timing_sweep.py`)
2. **Correlate Traces** - Match kernel traces with benchmarks (`correlate_rocprof_timings.py`) 
3. **Statistical Analysis** - Generate timing statistics (`analyze_timing_stats.py`)
4. **BIC Segmentation** - Identify performance segments (`segment_performance_bic.py`)
5. **Interactive Plot** - Create size vs time visualization (`plot_size_vs_time_plotly.py`)
6. **Boxplots** - Generate distribution plots per segment (`create_boxplots.py`)

**Note:** `plot_kernel_timeline.py` is not included in the pipeline (needs optimization). You can run it manually if needed:
```bash
python3 plot_kernel_timeline.py <run_dir>
```

## Quick Start

### Basic Usage

Run complete pipeline with defaults (8 ranks, 100 iterations):

```bash
cd /work/lmeadows/rccl/rccl-tests/scripts
python3 run_full_pipeline.py all_reduce
```

### Custom Configuration

```bash
# 4 ranks, 50 iterations
python3 run_full_pipeline.py all_reduce --ranks 4 --iterations 50

# Custom size range
python3 run_full_pipeline.py reduce_scatter --min-size 1024 --max-size 512M

# More warmup iterations
python3 run_full_pipeline.py all_gather --warmup 10
```

### Analyze Existing Data

If you already have a run directory and only want to regenerate analysis/visualizations:

```bash
python3 run_full_pipeline.py all_reduce --analyze-only \
    --run-dir /work/lmeadows/rccl/data/hostname/run_all_reduce_20251108_123456
```

### Skip Visualizations

For faster execution (useful for quick analysis):

```bash
python3 run_full_pipeline.py all_reduce --no-viz
```

## Command-Line Options

### Required Arguments

| Argument | Description |
|----------|-------------|
| `benchmark` | Benchmark name (e.g., all_reduce, all_gather, reduce_scatter) |

### Benchmark Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--ranks N` | 8 | Number of MPI ranks |
| `--iterations N` | 100 | Number of timed iterations per size |
| `--warmup N` | 5 | Number of warmup iterations |
| `--min-size SIZE` | auto | Minimum message size (auto-adjusted for alignment) |
| `--max-size SIZE` | 1G | Maximum message size |

### Pipeline Control

| Option | Description |
|--------|-------------|
| `--analyze-only` | Skip benchmark run, only analyze existing data |
| `--run-dir PATH` | Run directory (required with `--analyze-only`) |
| `--no-viz` | Skip visualization steps (faster) |
| `--continue-on-error` | Continue pipeline even if a step fails |

### Help

```bash
python3 run_full_pipeline.py --help
```

## Output Files

The pipeline generates these files in the run directory:

### Data Files
- `all_rank*.csv` - Individual kernel timings per rank
- `{benchmark}_benchmark_output.txt` - Raw benchmark output
- `{benchmark}_timing_analysis.csv` - Statistical summary
- `{benchmark}_bic_segmentation.json` - Segment boundaries and models
- `run_metadata.json` - Run configuration metadata

### Visualizations
- `{benchmark}_size_vs_time.html` - Interactive Plotly chart
- `{benchmark}_segment*_boxplots.png` - Boxplots per segment

### ROCProfiler Data
- `rocp/{hostname}/{pid}_kernel_trace.csv` - Kernel traces per rank
- `rank_{N}_timestamps.txt` - Benchmark timing markers per rank
- `rank_pid_mapping.json` - Rank to PID mapping

## Examples

### Example 1: Standard Run

```bash
python3 run_full_pipeline.py all_reduce
```

**Output:**
```
================================================================================
RCCL Performance Analysis Pipeline
================================================================================
Benchmark: all_reduce
Ranks: 8
Iterations: 100
================================================================================

================================================================================
Running: Step 1: Running Benchmark with Timing Sweep
...
✅ Benchmark completed successfully

✅ Found run directory: /work/lmeadows/rccl/data/hostname/run_all_reduce_20251108_123456

✅ Correlation already completed (8 rank files found)

================================================================================
Running: Step 3: Statistical Analysis
...
✅ Statistical Analysis completed successfully

================================================================================
Running: Step 4: BIC Segmentation
...
✅ BIC Segmentation completed successfully

================================================================================
Running: Step 5: Interactive Size vs Time Plot
...
✅ Interactive Size vs Time Plot completed successfully

================================================================================
Running: Step 6: Boxplot Visualizations
...
✅ Boxplot Visualizations completed successfully

================================================================================
Generated Outputs
================================================================================

Timing CSVs:
  ✓ all_rank0.csv (45.2 KB)
  ✓ all_rank1.csv (45.1 KB)
  ...

Statistical Analysis:
  ✓ all_reduce_timing_analysis.csv (3.2 KB)

BIC Segmentation:
  ✓ all_reduce_bic_segmentation.json (1.5 KB)

Interactive Plot:
  ✓ all_reduce_size_vs_time.html (856 KB)

Boxplots:
  ✓ all_reduce_segment0_boxplots.png (234 KB)
  ✓ all_reduce_segment1_boxplots.png (189 KB)
  ✓ all_reduce_segment2_boxplots.png (156 KB)

Total execution time: 382.5 seconds

================================================================================
Pipeline Execution Summary
================================================================================
Benchmark: all_reduce
Run Directory: /work/lmeadows/rccl/data/hostname/run_all_reduce_20251108_123456

Results:
  ✅ Successful steps: 6
  ❌ Failed steps: 0
  📁 Output files: 13

🎉 Pipeline completed successfully!

View results:
  cd /work/lmeadows/rccl/data/hostname/run_all_reduce_20251108_123456

  Open interactive visualizations:
    firefox all_reduce_size_vs_time.html
================================================================================
```

### Example 2: Re-analyze Existing Data

```bash
python3 run_full_pipeline.py all_reduce --analyze-only \
    --run-dir /work/lmeadows/rccl/data/hostname/run_all_reduce_20251108_123456
```

This will:
- Skip the benchmark run
- Regenerate all analysis and visualizations
- Useful if you want to try different analysis parameters

### Example 3: Quick Analysis (No Visualizations)

```bash
python3 run_full_pipeline.py reduce_scatter --ranks 4 --no-viz
```

This will:
- Run benchmark with 4 ranks
- Generate statistical analysis and segmentation
- Skip visualization steps (saves ~2-3 minutes)

### Example 4: Continue on Errors

```bash
python3 run_full_pipeline.py all_gather --continue-on-error
```

This will:
- Continue pipeline even if a step fails
- Useful for debugging
- Generates as many outputs as possible

## Available Benchmarks

- `all_reduce` - AllReduce collective
- `all_gather` - AllGather collective
- `broadcast` - Broadcast collective
- `reduce` - Reduce collective
- `reduce_scatter` - ReduceScatter collective
- `scatter` - Scatter collective
- `gather` - Gather collective
- `alltoall` - AllToAll collective
- `alltoallv` - AllToAllV collective
- `hypercube` - Hypercube collective
- `sendrecv` - Send/Receive point-to-point

## Execution Time

Typical execution times (8 ranks, 100 iterations, 8B-1GiB sweep):

| Step | Time |
|------|------|
| Benchmark Run | 5-8 minutes |
| Correlation | < 5 seconds |
| Statistical Analysis | < 2 seconds |
| BIC Segmentation | < 2 seconds |
| Interactive Plot | 1-2 seconds |
| Boxplots | 2-3 seconds |
| **Total** | **~6-10 minutes** |

## Troubleshooting

### Error: "No run directories found"

**Cause:** Benchmark didn't complete or data directory doesn't exist

**Solution:**
1. Check if benchmark completed successfully
2. Verify data directory exists: `/work/lmeadows/rccl/data/$(hostname)/`
3. Check for run_* directories in data directory

### Error: "No timing files found"

**Cause:** Correlation step failed

**Solution:**
1. Check if ROCProfiler traces exist in `rocp/` subdirectory
2. Verify timestamp files exist: `rank_*_timestamps.txt`
3. Try running `correlate_rocprof_timings.py` manually

### Error: "ModuleNotFoundError"

**Cause:** Missing Python packages

**Solution:**
```bash
pip install --user -r requirements.txt
```

### Visualization Fails

**Cause:** Missing matplotlib, seaborn, or plotly

**Solution:**
```bash
# For boxplots
pip install --user matplotlib seaborn

# For interactive plots
pip install --user plotly
```

## Integration with Existing Workflow

### Manual Steps (Old Way)

```bash
# Step 1
python3 run_timing_sweep.py all_reduce --ranks 8

# Step 2
RUN_DIR=$(ls -td /work/lmeadows/rccl/data/$(hostname)/run_all_reduce_* | head -1)

# Step 3
python3 analyze_timing_stats.py $RUN_DIR

# Step 4
python3 segment_performance_bic.py $RUN_DIR

# Step 5
python3 plot_size_vs_time_plotly.py $RUN_DIR

# Step 6
python3 create_boxplots.py $RUN_DIR
```

### Automated (New Way)

```bash
python3 run_full_pipeline.py all_reduce --ranks 8
```

Both produce identical results, but the pipeline script:
- ✅ Ensures correct execution order
- ✅ Handles errors gracefully
- ✅ Tracks progress
- ✅ Summarizes outputs
- ✅ Saves time

## See Also

- `README.md` - Project overview and documentation
- `SCRIPT_DEPENDENCIES.md` - Package requirements
- `ANALYZE_TIMING_STATS.md` - Statistical analysis details
- `SEGMENT_PERFORMANCE_BIC.md` - Segmentation algorithm
- `AI_GUIDELINES.md` - Project conventions

---

**Last Updated:** November 8, 2025

