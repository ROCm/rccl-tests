# RCCL Performance Analysis - Quick Start

## One-Command Complete Analysis

```bash
cd /work/lmeadows/rccl/rccl-tests/scripts
python3 run_full_pipeline.py all_reduce
```

This single command will:
1. ✅ Run the benchmark with ROCProfiler tracing
2. ✅ Correlate kernel timings
3. ✅ Generate statistical analysis
4. ✅ Perform BIC segmentation
5. ✅ Create interactive visualizations
6. ✅ Generate boxplots

**Time:** ~6-10 minutes for complete pipeline

**Output:** 13+ files including interactive HTML visualizations

**Note:** `plot_kernel_timeline.py` excluded (needs optimization). Run manually if needed.

---

## Common Workflows

### Standard Benchmark Run
```bash
python3 run_full_pipeline.py all_reduce
```

### Custom Configuration
```bash
python3 run_full_pipeline.py all_reduce --ranks 4 --iterations 50
```

### Re-analyze Existing Data
```bash
python3 run_full_pipeline.py all_reduce --analyze-only \
    --run-dir /work/lmeadows/rccl/data/hostname/run_all_reduce_*
```

### Quick Analysis (Skip Visualizations)
```bash
python3 run_full_pipeline.py all_reduce --no-viz
```

---

## Available Benchmarks

| Benchmark | Description |
|-----------|-------------|
| `all_reduce` | AllReduce collective |
| `all_gather` | AllGather collective |
| `reduce_scatter` | ReduceScatter collective |
| `broadcast` | Broadcast collective |
| `reduce` | Reduce collective |
| `scatter` | Scatter collective |
| `gather` | Gather collective |
| `alltoall` | AllToAll collective |
| `sendrecv` | Send/Receive point-to-point |

---

## Install Dependencies

```bash
pip install --user -r requirements.txt
```

Required packages: pandas, numpy, scipy, matplotlib, seaborn, plotly

---

## View Results

After pipeline completes:

```bash
# Go to output directory
cd /work/lmeadows/rccl/data/$(hostname)/run_all_reduce_*/

# View interactive plot
firefox all_reduce_size_vs_time.html

# View boxplots
eog all_reduce_segment*_boxplots.png
```

---

## Help

```bash
python3 run_full_pipeline.py --help
```

---

## Documentation

- **Complete Guide:** `docs/PIPELINE_USAGE.md`
- **Script Reference:** `docs/README.md`
- **Dependencies:** `docs/SCRIPT_DEPENDENCIES.md`

---

## Troubleshooting

**Missing packages?**
```bash
pip install --user -r requirements.txt
```

**Need help?**
```bash
python3 run_full_pipeline.py --help
```

**Check script documentation:**
```bash
cat docs/PIPELINE_USAGE.md
```

