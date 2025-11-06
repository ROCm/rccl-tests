# Data Presentation Options for Individual Kernel Timings

This document records three recommended ways to present and analyze data generated when `--save_individual_timings` is used. By default, individual timing samples from all ranks should be aggregated into a single sample set for each message size and operation mode (in-place vs out-of-place).

## 1) Statistical Summary Tables

- Purpose: Provide concise, high-signal metrics per message size and operation.
- Aggregation: Aggregate samples across all ranks by default (configurable).
- Include wall-clock timing from benchmark output (csv/txt) alongside kernel timing statistics.
- Suggested columns per row (one row = one size × operation):
  - size_bytes, operation (in-place | out-of-place)
  - kernel_count (number of samples)
  - kernel_mean_us, kernel_std_us, kernel_min_us, kernel_max_us
  - kernel_p25_us, kernel_p50_us, kernel_p75_us, kernel_p95_us, kernel_p99_us
  - kernel_cv_percent (std/mean × 100)
  - wall_time_us (from benchmark output), optional: wall_errors
- Notes:
  - Filter/annotate warmup or zero-size entries as needed.
  - Keep wall times and kernel times separate to highlight CPU–GPU sync costs.

## 2) Distribution Visualization (Box Plots)

- Purpose: Show distribution and variability across iterations per size.
- Plot:
  - X-axis: message size (log scale recommended: 8 B → 1 GiB)
  - Y-axis: latency (μs)
  - Box shows IQR (P25–P75), median line (P50), whiskers/outliers
  - Color/facet by operation (in-place vs out-of-place); facet by rank if needed
- Insights:
  - Highlights iteration variance, outliers, and stability regions.
  - Useful for detecting jitter, scheduler effects, and tail latency.

## 3) Performance Scaling Analysis (Line Plots with Error Bars)

- Purpose: Characterize scaling with message size and compare modes/types.
- Plots:
  - Latency vs size (μs) with error bars (std or confidence interval)
  - Bandwidth vs size (GB/s) to show throughput scaling
  - Distinguish operation (in-place/out-of-place), data type, or rank count by color/line style
- Insights:
  - Reveals performance regimes, cross-over points, and scaling trends.
  - Error bars quantify measurement variability.

### Reference Plot Scripts

- Static PNG (Matplotlib): `/work/lmeadows/rccl/scripts/plot_size_vs_time.py`
  - Kernel mean ± std as error bars, wall-clock as points, X=Size (log2), Y=Time (log)
  - Output written to the run directory as `{benchmark}_size_vs_time.png`

- Interactive HTML (Plotly): `/work/lmeadows/rccl/scripts/plot_size_vs_time_plotly.py`
  - Interactive hover/zoom, shaded ±1σ band and error bars, same inputs/axes
  - Output written to the run directory as `{benchmark}_size_vs_time.html`

## Reusable Data Input

- Timing CSVs: `*_rankN.csv` aggregated across ranks to form the sample set per size/operation.
- Benchmark Output: parse `{benchmark}_benchmark_output.txt` for wall-clock time per size.
- Reference script (example implementation):
  - `/work/lmeadows/rccl/scripts/analyze_timing_stats.py`
  - Loads timing CSVs and the benchmark output, aggregates stats, correlates wall times.
  - Plotting scripts above consume the same inputs directly from a run directory.

## Implementation Notes

- Prefer log-scale on size axes (8 B → 1 GiB powers of two).
- Treat zero-size entries as warmup/initialization artifacts; filter or annotate.
- Keep kernel vs wall times separate; avoid conflating microsecond-level GPU timings with application-level wall times.

## Saved Summary: Segmented Size–Time Analysis

**Background**: We partition the curve using top = max(wall_time_us, kernel_mean_us + kernel_std_us), aggregating individual timings across ranks per size.

- **Segments**
  - Segment 0: 8–512 B
  - Segment 1: 1 KiB–256 KiB
  - Segment 2: 512 KiB–1 GiB

- **Best-fit formulas (y_us = a + b·size_MB)**
  - S0: 47.15 + 9706.21·size_MB (R²≈0.13) — overhead/noise-dominated
  - S1: 42.37 + 40.46·size_MB (R²≈0.56) — transitional regime
  - S2: 60.16 + 4.82·size_MB (R²≈1.00) — bandwidth-limited

- **Notes**
  - Clear break at 512 B; first exceed of prior max top at 512 KiB.
  - Large-size scaling is clean and near-linear; small sizes show higher variance.
  - Follow-ups: investigate S1’s slope (protocol/algorithm changes, runtime effects) and reduce small-size variance (warmups, binding, MPI settings).

