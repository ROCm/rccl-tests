# Script Dependencies Reference

This document shows which Python packages are required by each script.

## Quick Install

Install all dependencies:
```bash
pip install -r requirements.txt
```

Or with user install (no sudo required):
```bash
pip install --user -r requirements.txt
```

---

## Dependency Matrix

| Script | pandas | numpy | scipy | matplotlib | seaborn | plotly |
|--------|--------|-------|-------|------------|---------|--------|
| **run_timing_sweep.py** | - | - | - | - | - | - |
| **correlate_rocprof_timings.py** | ✓ | - | - | - | - | - |
| **analyze_timing_stats.py** | ✓ | ✓ | - | - | - | - |
| **segment_performance_bic.py** | ✓ | ✓ | ✓ | - | - | - |
| **create_boxplots.py** | ✓ | ✓ | - | ✓ | ✓ | - |
| **plot_size_vs_time_plotly.py** | ✓ | ✓ | - | - | - | ✓ |
| **plot_kernel_timeline.py** | - | - | - | - | - | ✓ |
| **analyze_nccl_kernel_counts.py** | ✓ | - | - | - | - | - |
| **compare_rccL_gpu.py** | ✓ | ✓ | - | ✓ | - | - |
| **visualize_gpu_benchmark.py** | ✓ | ✓ | - | ✓ | - | - |

---

## Package Usage Details

### pandas (Required by most scripts)
**Used by:** correlate_rocprof_timings.py, analyze_timing_stats.py, segment_performance_bic.py, create_boxplots.py, plot_size_vs_time_plotly.py, analyze_nccl_kernel_counts.py, compare_rccL_gpu.py, visualize_gpu_benchmark.py

**Purpose:** CSV file loading and data manipulation

**Minimum version:** 2.0.0

---

### numpy (Required by analysis and visualization scripts)
**Used by:** analyze_timing_stats.py, segment_performance_bic.py, create_boxplots.py, plot_size_vs_time_plotly.py, compare_rccL_gpu.py, visualize_gpu_benchmark.py

**Purpose:** Numerical operations, statistics, array manipulation

**Minimum version:** 1.24.0

---

### scipy (Required for BIC segmentation)
**Used by:** segment_performance_bic.py

**Purpose:** Statistical functions, curve fitting, optimization

**Minimum version:** 1.10.0

**Note:** Only needed if you use `segment_performance_bic.py`

---

### matplotlib (Required for static plots)
**Used by:** create_boxplots.py, compare_rccL_gpu.py, visualize_gpu_benchmark.py

**Purpose:** Creating PNG boxplots and static visualizations

**Minimum version:** 3.7.0

**Note:** Not needed if you only use Plotly (interactive) visualizations

---

### seaborn (Required for enhanced boxplots)
**Used by:** create_boxplots.py

**Purpose:** Statistical data visualization, enhanced boxplot styling

**Minimum version:** 0.12.0

**Note:** Only needed for `create_boxplots.py`

---

### plotly (Required for interactive visualizations)
**Used by:** plot_size_vs_time_plotly.py, plot_kernel_timeline.py

**Purpose:** Interactive HTML visualizations with zoom/pan/hover

**Minimum version:** 5.14.0

**Note:** Not needed if you only use matplotlib (static) visualizations

---

## Minimal Installation Options

### Core Pipeline Only (no visualization)
```bash
pip install pandas>=2.0.0 numpy>=1.24.0
```

**Enables:**
- run_timing_sweep.py (no deps)
- correlate_rocprof_timings.py
- analyze_timing_stats.py

---

### With BIC Segmentation
```bash
pip install pandas>=2.0.0 numpy>=1.24.0 scipy>=1.10.0
```

**Adds:**
- segment_performance_bic.py

---

### With Static Visualization
```bash
pip install pandas>=2.0.0 numpy>=1.24.0 matplotlib>=3.7.0 seaborn>=0.12.0
```

**Adds:**
- create_boxplots.py
- compare_rccL_gpu.py
- visualize_gpu_benchmark.py

---

### With Interactive Visualization
```bash
pip install pandas>=2.0.0 numpy>=1.24.0 plotly>=5.14.0
```

**Adds:**
- plot_size_vs_time_plotly.py
- plot_kernel_timeline.py

---

## Python Version

**Required:** Python 3.10 or newer

**Tested with:** Python 3.10.12

Check your Python version:
```bash
python3 --version
```

---

## Standard Library Modules

These modules are included with Python and require no installation:
- os, sys, re, json, glob, csv
- argparse, socket, pathlib, collections
- datetime, shutil, subprocess

---

## Troubleshooting

### Import Error

If you see:
```
ModuleNotFoundError: No module named 'pandas'
```

Install the missing package:
```bash
pip install --user pandas
```

Or install all dependencies:
```bash
pip install --user -r requirements.txt
```

---

### Permission Errors

If `pip install` fails with permission errors, use `--user`:
```bash
pip install --user -r requirements.txt
```

This installs packages in your home directory (`~/.local/lib/python3.10/site-packages/`)

---

### Virtual Environment (Recommended)

For isolated package management:
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install packages
pip install -r requirements.txt

# When done
deactivate
```

---

## Version Information

**Last Updated:** November 7, 2025

**Tested Package Versions:**
- pandas: 2.2.3
- numpy: 2.2.0
- scipy: 1.15.3
- matplotlib: 3.10.7
- seaborn: 0.13.2
- plotly: 6.4.0

