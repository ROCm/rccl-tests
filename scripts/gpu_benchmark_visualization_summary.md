# GPU Interconnect Benchmark Data Visualization

**Date:** October 31, 2025
**Hardware:** 4x AMD Instinct MI300A GPUs
**Test:** Raw GPU-to-GPU transfer performance

---

## Data Overview

- **Total measurements:** 372 data points
- **Latency tests:** 120 measurements (8B-4KB, 12 GPU pairs)
- **Bandwidth tests:** 252 measurements (1KB-1GB, 12 GPU pairs)
- **GPU pairs:** All 12 combinations tested bidirectionally

---

## Key Findings from Visualizations

### Latency Performance
- **Range:** 7.17 - 9.99 μs across all measurements
- **Average:** 8.84 μs
- **Variation:** 22.1% between best/worst GPU pairs
- **Best pair:** 0→3 (7.46 μs average)
- **Worst pair:** 2→0 (9.42 μs average)

### Bandwidth Performance
- **Range:** 0.64 - 435.66 GB/s across all measurements
- **Average:** 247.68 GB/s
- **Peak:** 436 GB/s (hardware limit reached)
- **Variation:** 5.1% between best/worst GPU pairs
- **Best pair:** 0→2 (350.35 GB/s average)
- **Worst pair:** 2→0 (332.72 GB/s average)

---

## Visualization Types

### Detailed Plots (`gpu_benchmark_detailed.png`)
1. **Latency vs Message Size:** All 12 GPU pairs, 8B-4KB
2. **Latency Distribution:** Box plots showing spread across pairs
3. **Average Latency by Pair:** Ranked performance comparison
4. **Bandwidth vs Message Size:** All 12 GPU pairs, 1KB-1GB
5. **Bandwidth Distribution:** Box plots for large messages (64KB+)
6. **Average Bandwidth by Pair:** Ranked performance comparison

### Summary Plots (`gpu_benchmark_summary.png`)
1. **Latency Statistics:** Mean ± std across message sizes
2. **Bandwidth Statistics:** Mean ± std across message sizes
3. **GPU Pair Latency Comparison:** Performance ranking
4. **GPU Pair Bandwidth Comparison:** Performance ranking

---

## Performance Insights

### Latency Characteristics
- **Very low baseline:** ~8 μs minimum across all pairs
- **Consistent scaling:** Latency increases minimally with message size
- **Pair variation:** Some GPU pairs have better interconnect paths

### Bandwidth Characteristics
- **Excellent scaling:** From 0.64 GB/s (1KB) to 436 GB/s (1GB)
- **Hardware saturation:** Clear plateau at ~436 GB/s for large messages
- **Low variation:** Only 5.1% difference between best/worst pairs

### Hardware Architecture Insights
- **Symmetric performance:** Most GPU pairs perform similarly
- **High bandwidth:** 436 GB/s sustained is world-class
- **Low latency:** 8-10 μs is excellent for GPU interconnects
- **Consistent quality:** All pairs maintain high performance

---

## Files Generated

- `visualize_gpu_benchmark.py` - Visualization script
- `gpu_benchmark_detailed.png` - Comprehensive detailed plots
- `gpu_benchmark_summary.png` - Summary statistics plots
- `gpu_transfer_results.txt` - Raw benchmark data

---

## Web Access

- **Detailed plots:** `http://localhost:8080/gpu_benchmark_detailed.png`
- **Summary plots:** `http://localhost:8080/gpu_benchmark_summary.png`

---

*These visualizations provide comprehensive insight into the raw GPU interconnect performance, showing excellent bandwidth scaling and consistent latency across all GPU pairs on the MI300A platform.*
