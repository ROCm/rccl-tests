# segment_performance_bic.py - Documentation

## Overview

`segment_performance_bic.py` is an automated performance segmentation tool that divides RCCL benchmark data into exactly **3 performance segments** using Bayesian Information Criterion (BIC). It identifies optimal breakpoints that separate latency-dominated, transition, and bandwidth-dominated performance regimes.

**Location:** `/work/lmeadows/rccl/scripts/segment_performance_bic.py`

**Version:** 1.0

**Algorithm:** Piecewise Linear Regression with BIC (Algorithm 2 from proposals)

**Author:** RCCL Performance Analysis Team

---

## Purpose

The script serves as the **second analysis step** in the RCCL performance analysis pipeline:

1. **Identify** optimal breakpoints using exhaustive search
2. **Segment** performance data into 3 physically meaningful regions
3. **Fit** linear or log-linear models to each segment
4. **Export** segmentation metadata for visualization

### Why BIC?

**Bayesian Information Criterion (BIC)** balances model fit quality against complexity:

```
BIC = n·log(RSS/n) + k·log(n)

where:
  n = number of data points
  k = number of parameters (2 per segment for linear/log-linear)
  RSS = residual sum of squares
```

**Lower BIC = Better model**

BIC penalizes overfitting, ensuring we don't create unnecessary segments.

---

## Algorithm

### Piecewise Linear Regression with BIC

**Objective:** Find 2 breakpoints that minimize BIC for 3 segments

**Method:** Exhaustive search

**Steps:**

1. **Load Data:**
   - Read benchmark output for wall-clock times
   - Extract sizes and times (out-of-place operation)
   - Filter out zero-size entries

2. **Exhaustive Search:**
   - Try all possible pairs of breakpoints (i, j) where:
     - i ∈ [2, n-4] (first breakpoint)
     - j ∈ [i+2, n-2] (second breakpoint)
     - Ensures minimum 2 points per segment
   - For each pair, compute BIC
   - Select pair with minimum BIC

3. **Model Fitting:**
   - For each of 3 segments:
     - Fit linear model: `y = a + b·size_MB`
     - Fit log-linear model: `y = a + b·log2(size)`
     - Choose model with better R²
     - Record formula, R², and parameters

4. **Export Results:**
   - Save segmentation to JSON
   - Include breakpoints, formulas, R² values

---

## Input Requirements

### Directory Structure

The script expects a run directory created by `run_timing_sweep.py`:

```
/work/lmeadows/rccl/data/<hostname>/run_{benchmark}_{YYYYMMDD_HHMMSS}/
├── {benchmark}_benchmark_output.txt   # Required: Wall-clock times
├── {benchmark}_rank*.csv              # Optional: For timing data
├── run_metadata.json                  # Metadata
└── sweep_summary.txt                  # Summary
```

### Input File Format

**Benchmark Output (`*_benchmark_output.txt`):**

The script parses timing lines:
```
#       size         count      type   redop    root     time   algbw   busbw #wrong
        1024           256     float     sum      -1    37.15    0.03    0.05      0
        2048           512     float     sum      -1    38.42    0.05    0.09      0
```

**Data Used:**
- Column 1: `size_bytes` (message size)
- Column 6: `time_us` (wall-clock time in microseconds)

---

## Usage

### Basic Usage

```bash
python3 segment_performance_bic.py --run-dir <run_directory>
```

### Examples

**Segment a specific run:**
```bash
python3 segment_performance_bic.py \
    --run-dir /work/lmeadows/rccl/data/cv350-zts-gtu-e11-18/run_all_reduce_20251104_165031
```

**Segment latest run:**
```bash
latest=$(ls -1td /work/lmeadows/rccl/data/$(hostname)/run_all_reduce_* | head -1)
python3 segment_performance_bic.py --run-dir "$latest"
```

**Batch segmentation:**
```bash
for dir in /work/lmeadows/rccl/data/$(hostname)/run_*_20251104_165*/; do
    python3 segment_performance_bic.py --run-dir "$dir"
done
```

---

## Output

### Console Output

```
Segmenting all_reduce benchmark data from run_all_reduce_20251104_165031
Using: Piecewise Linear Regression with BIC (3 segments)
================================================================================
Parsed 28 timing entries from benchmark output
Data points: 28
Size range: 8 to 1073741824 bytes

Searching for optimal 2 breakpoints among 28 data points...
Optimal breakpoints found at indices: [20, 25]
BIC = 90.66

================================================================================
SEGMENTATION RESULTS
================================================================================
Benchmark: all_reduce
Algorithm: Piecewise Linear Regression with BIC
BIC: 90.66
Breakpoints at sizes: 8388608, 268435456 bytes

Segment 0: 8..4194304 bytes
  Model: linear
  Formula: y = 44.73 + 8.74·size_MB
  R² = 0.8112
  N = 20 points

Segment 1: 8388608..134217728 bytes
  Model: linear
  Formula: y = 70.13 + 4.73·size_MB
  R² = 0.9999
  N = 5 points

Segment 2: 268435456..1073741824 bytes
  Model: linear
  Formula: y = 49.40 + 4.82·size_MB
  R² = 1.0000
  N = 3 points

Results saved to: run_all_reduce_20251104_165031/all_reduce_segmentation_bic.json
```

### JSON Output

**File:** `{benchmark}_segmentation_bic.json`

**Location:** Same directory as input files

**Structure:**

```json
{
  "benchmark": "all_reduce",
  "algorithm": "piecewise_linear_bic",
  "n_segments": 3,
  "n_datapoints": 28,
  "bic": 90.66,
  "breakpoint_indices": [20, 25],
  "breakpoint_sizes": [8388608, 268435456],
  "segments": [
    {
      "segment": 0,
      "size_range_bytes": [8, 4194304],
      "index_range": [0, 19],
      "n_points": 20,
      "model": "linear",
      "formula": "y = 44.73 + 8.74·size_MB",
      "parameters": {
        "a": 44.73,
        "b": 8.74
      },
      "r_squared": 0.8112
    },
    {
      "segment": 1,
      "size_range_bytes": [8388608, 134217728],
      "index_range": [20, 24],
      "n_points": 5,
      "model": "linear",
      "formula": "y = 70.13 + 4.73·size_MB",
      "parameters": {
        "a": 70.13,
        "b": 4.73
      },
      "r_squared": 0.9999
    },
    {
      "segment": 2,
      "size_range_bytes": [268435456, 1073741824],
      "index_range": [25, 27],
      "n_points": 3,
      "model": "linear",
      "formula": "y = 49.40 + 4.82·size_MB",
      "parameters": {
        "a": 49.40,
        "b": 4.82
      },
      "r_squared": 1.0000
    }
  ]
}
```

---

## Functions

### Data Loading Functions

#### `load_timing_data(output_dir)`

Loads timing CSV files (optional, for compatibility).

**Parameters:**
- `output_dir` (str): Path to run directory

**Returns:**
- `pd.DataFrame` or `None`: Combined timing data

**Note:** This function is available but not required for BIC segmentation. The script primarily uses benchmark output.

---

#### `load_benchmark_output(output_dir, benchmark_name)`

Loads benchmark output text file.

**Parameters:**
- `output_dir` (str): Path to run directory
- `benchmark_name` (str): Benchmark name

**Returns:**
- `str` or `None`: Raw file content

---

#### `parse_benchmark_output(content)`

Parses benchmark output to extract wall-clock times.

**Parameters:**
- `content` (str): Raw benchmark output

**Returns:**
- `pd.DataFrame`: Columns: size_bytes, wall_time_oop_us, wall_time_ip_us

**Regex Pattern:**
```python
r'^\s*(\d+)\s+(\d+)\s+(\w+)\s+(\w+)\s+(-?\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)'
```

---

### Model Fitting Functions

#### `fit_linear(x, y)`

Fits linear model: `y = a + b·x` where x is in MB.

**Parameters:**
- `x` (array): Message sizes in bytes
- `y` (array): Times in microseconds

**Returns:**
- `dict` or `None`: Model information

**Dictionary Keys:**
- `model`: "linear"
- `formula`: Human-readable formula string
- `params`: {a, b} coefficients
- `r2`: R² goodness of fit
- `rss`: Residual sum of squares
- `n`: Number of data points

**Implementation:**
```python
x_mb = x / (1024 * 1024)  # Convert to MB
A = [ones, x_mb]
[a, b] = lstsq(A, y)
y_pred = a + b * x_mb
r2 = 1 - RSS/TSS
```

---

#### `fit_loglinear(x, y)`

Fits log-linear model: `y = a + b·log2(x)`.

**Parameters:**
- `x` (array): Message sizes in bytes
- `y` (array): Times in microseconds

**Returns:**
- `dict` or `None`: Model information

**Implementation:**
```python
log_x = log2(x)
A = [ones, log_x]
[a, b] = lstsq(A, y)
y_pred = a + b * log_x
r2 = 1 - RSS/TSS
```

---

#### `fit_segment(sizes, times)`

Fits both linear and log-linear models, returns best.

**Parameters:**
- `sizes` (array): Message sizes
- `times` (array): Times

**Returns:**
- `dict`: Best model (highest R²)

**Logic:**
```python
linear_fit = fit_linear(sizes, times)
loglin_fit = fit_loglinear(sizes, times)
return max(linear_fit, loglin_fit, key=lambda x: x['r2'])
```

---

### BIC Calculation Functions

#### `compute_bic_for_breaks(sizes, times, breaks)`

Computes BIC for a given breakpoint configuration.

**Parameters:**
- `sizes` (array): Message sizes
- `times` (array): Times
- `breaks` (list): Breakpoint indices [i, j]

**Returns:**
- `float`: BIC value (lower is better)

**Formula:**
```python
n = len(sizes)
k = 2 * (len(breaks) + 1)  # 2 params per segment
RSS = sum of residuals across all segments
BIC = n * log(RSS/n) + k * log(n)
```

**Edge Cases:**
- Returns `inf` if any segment has < 2 points
- Returns `inf` if RSS ≤ 0 (invalid fit)

---

#### `find_optimal_breakpoints(sizes, times, n_segments=3)`

Finds optimal breakpoints using exhaustive search.

**Parameters:**
- `sizes` (array): Message sizes
- `times` (array): Times
- `n_segments` (int): Number of segments (always 3)

**Returns:**
- `tuple`: (breakpoint_indices, segment_info, bic_value)

**Algorithm:**
```python
best_bic = infinity
for i in range(2, n-4):
    for j in range(i+2, n-2):
        bic = compute_bic_for_breaks(sizes, times, [i, j])
        if bic < best_bic:
            best_bic = bic
            best_breaks = [i, j]
```

**Complexity:** O(n²) where n is number of data points (~28)

**Constraints:**
- Minimum 2 points per segment
- Breakpoints must be ordered: 0 < i < j < n

---

### Main Function

#### `segment_benchmark_data(output_dir, benchmark_name)`

Main segmentation orchestrator.

**Parameters:**
- `output_dir` (str): Run directory
- `benchmark_name` (str): Benchmark name

**Returns:**
- `dict` or `None`: Segmentation results

**Process:**
1. Load benchmark output
2. Parse wall-clock times
3. Filter out zero sizes
4. Find optimal breakpoints
5. Fit models to segments
6. Format results
7. Return dictionary

---

## Statistical Measures

### BIC (Bayesian Information Criterion)

**Purpose:** Model selection criterion that penalizes complexity

**Formula:**
```
BIC = n·log(RSS/n) + k·log(n)
```

**Components:**
- `n`: Sample size (number of data points)
- `RSS`: Residual sum of squares (fit quality)
- `k`: Number of parameters (complexity penalty)

**Interpretation:**
- Lower BIC = Better model
- First term: Rewards good fit (low RSS)
- Second term: Penalizes complexity (high k)

**Comparison:**
```
Model A: BIC = 90.66  (3 segments, 6 parameters)
Model B: BIC = 120.45 (5 segments, 10 parameters)
→ Choose Model A (lower BIC)
```

---

### R² (Coefficient of Determination)

**Purpose:** Measures goodness of fit

**Formula:**
```
R² = 1 - (RSS / TSS)

where:
  RSS = Σ(y_actual - y_predicted)²
  TSS = Σ(y_actual - y_mean)²
```

**Interpretation:**
- R² = 1.0: Perfect fit
- R² = 0.9: 90% of variance explained
- R² = 0.5: 50% of variance explained
- R² < 0.5: Poor fit

**Typical Values:**
- Segment 0 (small): R² = 0.6-0.9 (latency-dominated, noisy)
- Segment 1 (medium): R² = 0.95-0.999 (transition)
- Segment 2 (large): R² = 0.999-1.0 (bandwidth-dominated, linear)

---

## Segment Interpretation

### 3-Segment Model

The algorithm produces exactly 3 segments that correspond to physical performance regimes:

#### Segment 0: Latency-Dominated (Small Messages)

**Typical Range:** 8 bytes to ~1 MiB

**Characteristics:**
- Flat or slowly increasing time
- High variability (CV > 20%)
- Often log-linear behavior
- R² = 0.6-0.9 (lower due to noise)

**Physics:**
- Fixed overhead dominates (kernel launch, synchronization)
- Message size has minimal impact
- Cache effects

**Model:** `y ≈ L₀ + c·log(size)` or `y ≈ L₀ + small·size`

**Optimization:** Reduce latency, batch small messages

---

#### Segment 1: Transition (Medium Messages)

**Typical Range:** ~1 MiB to ~64-256 MiB

**Characteristics:**
- Linear scaling with size
- Moderate slope
- R² = 0.95-0.999

**Physics:**
- Algorithm changes (e.g., ring → tree)
- Protocol transitions (eager → rendezvous)
- Cache effects diminishing

**Model:** `y ≈ a + b·size` (moderate b)

**Optimization:** Tune algorithm selection thresholds

---

#### Segment 2: Bandwidth-Dominated (Large Messages)

**Typical Range:** ~64-256 MiB to 1 GiB

**Characteristics:**
- Linear scaling with size
- Consistent slope
- R² ≈ 1.0 (near-perfect fit)

**Physics:**
- Network/memory bandwidth is bottleneck
- Time = Latency + Size/Bandwidth
- Predictable performance

**Model:** `y ≈ L₀ + size/B_eff`

**Optimization:** Maximize bandwidth utilization

---

## Breakpoint Patterns

### Common Breakpoint Locations

Based on analysis of 12 RCCL benchmarks:

| Transition | Typical Range | Physical Cause |
|------------|---------------|----------------|
| Small → Medium | 512 KiB - 2 MiB | Cache effects, protocol change |
| Medium → Large | 64 MiB - 256 MiB | Bandwidth-dominated regime |

### Benchmark-Specific Patterns

**Early Transitions (<1 MiB):**
- `scatter`: 1 KiB, 2 MiB
- `reduce_scatter`: 128 KiB, 64 MiB
- Reason: Algorithm-specific behavior

**Standard Transitions (~1 MiB, ~256 MiB):**
- `all_reduce`: 8 MiB, 256 MiB
- `broadcast`: 512 KiB, 64 MiB
- `reduce`: 1 MiB, 256 MiB
- Reason: Typical RCCL algorithm switching

---

## Error Handling

### Missing Benchmark Output

```
Error: Could not load benchmark output
```

**Cause:** `{benchmark}_benchmark_output.txt` not found

**Solution:** Verify `run_timing_sweep.py` completed successfully

---

### Insufficient Data Points

```
Error: Not enough data points (N) for 3-segment analysis
```

**Cause:** Fewer than 7 data points (need minimum 2 per segment + 1)

**Solution:** Run benchmark with more sizes or lower minimum size

---

### Invalid Breakpoints

```
ValueError: Could not find valid breakpoints
```

**Cause:** All breakpoint combinations result in invalid BIC (inf)

**Possible Reasons:**
- All sizes are zero
- Degenerate data (all same value)
- Numerical instability

**Solution:** Check benchmark output for valid data

---

## Integration with Pipeline

### Position in Analysis Pipeline

```
1. run_timing_sweep.py          → Generate raw data
2. analyze_timing_stats.py      → Statistical summary
3. segment_performance_bic.py   → Identify segments (YOU ARE HERE)
4. plot_size_vs_time_plotly.py  → Visualize with segment lines
```

### Downstream Consumers

**Scripts that use the JSON output:**
- `plot_size_vs_time_plotly.py` - Adds vertical lines at breakpoints
- `plot_segment_boxplots.py` - Creates per-segment boxplots
- Custom analysis scripts - Use segment boundaries for analysis

**Data Used:**
- `breakpoint_sizes` - For vertical lines on plots
- `segments[i].size_range_bytes` - For filtering data by segment
- `segments[i].formula` - For displaying fit equations
- `segments[i].r_squared` - For assessing fit quality

---

## Performance Considerations

### Computational Complexity

**Exhaustive Search:** O(n²)
- For n=28 data points: ~350 BIC evaluations
- Each BIC evaluation: 3 linear fits + RSS calculation
- Total time: 1-2 seconds

**Memory Usage:**
- Input data: < 1 KB (28 sizes × 2 operations)
- Working memory: < 1 MB
- Output JSON: < 2 KB

### Scalability

**Tested Configurations:**
- Data points: 7 to 100
- Message sizes: 8 B to 8 GiB
- Execution time: < 5 seconds for n=100

**Bottleneck:** Exhaustive search (O(n²))

**Alternative:** For n > 100, consider dynamic programming (O(n²) but faster constant)

---

## Comparison with Heuristic Algorithm

### Old Algorithm (`segment_performance.py`)

**Method:** Heuristic break detection
- Looks for drops in TOP values
- Uses R² refinement
- No formal optimization

**Results:**
- Variable segment count (3-5)
- Ad-hoc breakpoints
- Some poor fits (R² < 0.5)

---

### New Algorithm (`segment_performance_bic.py`)

**Method:** BIC optimization
- Exhaustive search for optimal breaks
- Formal statistical criterion
- Guaranteed 3 segments

**Results:**
- Exactly 3 segments (consistent)
- Optimal breakpoints (minimizes BIC)
- Better fits (most R² > 0.9)

---

### Comparison Example: all_reduce

**Old (Heuristic):**
- Segments: 5
- Breaks: 64, 8192, 65536, 262144 bytes
- R² range: 0.48 - 1.0 (inconsistent)

**New (BIC):**
- Segments: 3
- Breaks: 8388608, 268435456 bytes
- R² range: 0.81 - 1.0 (consistent)
- BIC: 90.66

**Improvement:**
- Simpler model (3 vs 5 segments)
- More interpretable breakpoints
- Statistically justified

---

## Troubleshooting

### Issue: BIC value is very large (>1000)

**Cause:** Poor fit quality (high RSS)

**Possible Reasons:**
- Data is highly non-linear
- Outliers present
- Wrong model family (linear/log-linear insufficient)

**Solution:**
- Check data quality
- Consider polynomial models (future enhancement)
- Inspect R² values per segment

---

### Issue: All segments have same breakpoints

**Cause:** Data is too uniform or too small

**Solution:**
- Increase size range
- Add more intermediate sizes
- Check if benchmark is working correctly

---

### Issue: Segment 0 has low R² (<0.7)

**Cause:** Latency-dominated regime is inherently noisy

**Effect:** This is expected and acceptable

**Interpretation:** Small messages have high relative variability due to fixed overheads

---

## Examples

### Example 1: Basic Segmentation

```bash
python3 segment_performance_bic.py \
    --run-dir /work/lmeadows/rccl/data/cv350-zts-gtu-e11-18/run_all_reduce_20251104_165031
```

**Output:**
```
Optimal breakpoints found at indices: [20, 25]
BIC = 90.66
Breakpoints at sizes: 8388608, 268435456 bytes

Segment 0: 8..4194304 bytes (8B-4MiB)
Segment 1: 8388608..134217728 bytes (8MiB-128MiB)
Segment 2: 268435456..1073741824 bytes (256MiB-1GiB)
```

---

### Example 2: Batch Segmentation

```bash
#!/bin/bash
DATA_DIR="/work/lmeadows/rccl/data/$(hostname)"

for run_dir in "$DATA_DIR"/run_*_20251104_165*/; do
    python3 segment_performance_bic.py --run-dir "$run_dir"
done
```

---

### Example 3: Extract Breakpoints with Python

```python
import json

# Load segmentation results
with open("run_all_reduce_20251104_165031/all_reduce_segmentation_bic.json") as f:
    seg = json.load(f)

# Extract breakpoints
breaks = seg['breakpoint_sizes']
print(f"Breakpoints: {breaks[0]} bytes, {breaks[1]} bytes")

# Convert to human-readable
def format_size(b):
    if b >= 1024**3: return f"{b/1024**3:.0f} GiB"
    if b >= 1024**2: return f"{b/1024**2:.0f} MiB"
    if b >= 1024: return f"{b/1024:.0f} KiB"
    return f"{b} B"

print(f"Breakpoints: {format_size(breaks[0])}, {format_size(breaks[1])}")
# Output: Breakpoints: 8 MiB, 256 MiB
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-04 | Initial implementation with BIC optimization |

---

## See Also

- **`SEGMENTATION_ALGORITHM_PROPOSALS.md`** - Algorithm theory and alternatives
- **`BIC_SEGMENTATION_TEST_RESULTS.txt`** - Testing and validation results
- **`analyze_timing_stats.py`** - Statistical analysis (prerequisite)
- **`plot_size_vs_time_plotly.py`** - Visualization with segments
- **`AI_GUIDELINES.md`** - Project guidelines

---

## References

1. Schwarz, G. (1978). "Estimating the dimension of a model". Annals of Statistics, 6(2): 461–464.
2. Muggeo, V. M. R. (2003). "Estimating regression models with unknown break-points". Statistics in Medicine, 22(19): 3055–3071.
3. Killick, R., Fearnhead, P., & Eckley, I. A. (2012). "Optimal detection of changepoints with a linear computational cost". Journal of the American Statistical Association, 107(500): 1590–1598.

---

## Contact

For issues or questions about this script, refer to the RCCL testing project documentation.


