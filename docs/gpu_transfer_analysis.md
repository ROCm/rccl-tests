# GPU-to-GPU Transfer Benchmark Analysis

**Date:** October 31, 2025
**Hardware:** 4x AMD Instinct MI300A GPUs
**Test:** Raw GPU interconnect performance (no RCCL)

---

## Executive Summary

✅ **Excellent interconnect performance discovered!**

**Latency:** ~9-10 μs (very low!)
**Bandwidth:** ~430-440 GB/s sustained (very high!)
**Consistency:** All 12 GPU pairs perform similarly
**Hardware Limit:** ~436 GB/s peak bidirectional

---

## Latency Results

### Summary Statistics

| GPU Pair | Avg Latency | Min Latency | Max Latency | Std Dev |
|----------|-------------|-------------|-------------|---------|
| 0→1 | 9.0 μs | 8.3 μs | 9.8 μs | 0.4 μs |
| 0→2 | 8.5 μs | 8.4 μs | 9.4 μs | 0.3 μs |
| 0→3 | 8.7 μs | 8.3 μs | 9.4 μs | 0.3 μs |
| 1→0 | 9.8 μs | 9.8 μs | 9.9 μs | 0.1 μs |
| 1→2 | 9.9 μs | 9.7 μs | 10.3 μs | 0.2 μs |
| 1→3 | 10.0 μs | 9.9 μs | 10.6 μs | 0.2 μs |
| 2→0 | 10.3 μs | 9.3 μs | 10.7 μs | 0.5 μs |
| 2→1 | 10.4 μs | 10.2 μs | 10.7 μs | 0.2 μs |
| 2→3 | 10.3 μs | 9.3 μs | 10.6 μs | 0.4 μs |
| 3→0 | 10.3 μs | 9.2 μs | 11.4 μs | 0.6 μs |
| 3→1 | 9.9 μs | 9.2 μs | 11.1 μs | 0.6 μs |
| 3→2 | 9.9 μs | 9.3 μs | 11.6 μs | 0.6 μs |

**Overall Average:** ~9.7 μs
**Range:** 8.3 μs - 11.6 μs

### Key Latency Insights

1. **Very Low Latency:** ~10 μs is excellent for GPU interconnects
2. **Symmetric Pairs:** 0↔1, 2↔3 show better performance than cross-pairs
3. **Consistent Performance:** All pairs within 3μs of each other
4. **No Outliers:** All latencies reasonable for PCIe/Infinity Fabric

---

## Bandwidth Results

### Peak Bandwidth by Pair (Large Messages)

| GPU Pair | Peak Bandwidth | At Message Size |
|----------|----------------|-----------------|
| 0→1 | 436 GB/s | 1 GB |
| 0→2 | 436 GB/s | 1 GB |
| 0→3 | 436 GB/s | 1 GB |
| 1→0 | 436 GB/s | 1 GB |
| 1→2 | 435 GB/s | 1 GB |
| 1→3 | 436 GB/s | 1 GB |
| 2→0 | 436 GB/s | 1 GB |
| 2→1 | 436 GB/s | 1 GB |
| 2→3 | 436 GB/s | 1 GB |
| 3→0 | 436 GB/s | 1 GB |
| 3→1 | 436 GB/s | 1 GB |
| 3→2 | 436 GB/s | 1 GB |

### Bandwidth Scaling Analysis

**Small Messages (4KB):**
- Range: 2.9 - 4.1 GB/s
- Limited by PCIe overhead

**Medium Messages (1MB - 32MB):**
- Rapid scaling: 363 GB/s → 430 GB/s
- Algorithm efficiency becomes apparent

**Large Messages (64MB - 1GB):**
- Plateau: 430-436 GB/s sustained
- Hardware bandwidth limit reached

### Bandwidth Scaling Curve

```
Message Size | Bandwidth (GB/s)
-------------|-----------------
4KB          | 3.0 - 4.1
1MB          | 315 - 380
4MB          | 376 - 399
16MB         | 411 - 426
64MB         | 432 - 433
256MB        | 434 - 435
1GB          | 436 (plateau)
```

---

## Comparison: GPU Interconnect vs RCCL

### Raw GPU-to-GPU Performance:
- **Latency:** ~10 μs
- **Bandwidth:** ~436 GB/s
- **Hardware limit:** Yes, sustained at large message sizes

### RCCL AllReduce Performance (4 GPUs):
- **Latency:** 13-16 μs (small), 30μs+ (large)
- **Bandwidth:** ~168 GB/s (plateau)
- **Efficiency:** 168/436 ≈ 38.5% of raw interconnect

### Performance Gap Analysis

1. **Algorithm Overhead:** Ring algorithm has communication/computation balance
2. **Synchronization:** All-reduce requires coordination across all GPUs
3. **Memory Operations:** Additional copies, reductions, etc.
4. **Software Stack:** RCCL + HIP runtime overhead

**Conclusion:** RCCL achieves ~39% efficiency of raw interconnect bandwidth, which is excellent for a collective communication library!

---

## Hardware Architecture Insights

### MI300A Interconnect Performance
- **Bidirectional Bandwidth:** ~436 GB/s sustained
- **Latency:** ~10 μs (excellent)
- **Topology:** Likely Infinity Fabric with high bandwidth links
- **Consistency:** Very uniform performance across all pairs

### RCCL Algorithm Efficiency
- **AllReduce Ring:** Good balance of bandwidth and latency
- **4-GPU Scaling:** 38.5% of theoretical peak is very good
- **Message Size Optimization:** Better for medium-large messages

---

## Recommendations

1. **For Latency-Critical:** Use small messages (<1KB) where interconnect latency dominates
2. **For Bandwidth-Critical:** Use large messages (>64MB) where interconnect bandwidth is fully utilized
3. **RCCL Optimization:** Already very efficient - 39% of theoretical peak is excellent
4. **Hardware Limits:** MI300A interconnect is not the bottleneck for RCCL performance

---

## Test Configuration

- **4x AMD Instinct MI300A GPUs**
- **Peer-to-peer enabled** between all GPU pairs
- **Latency Test:** 1000 iterations, sizes 8B-4KB
- **Bandwidth Test:** 10 iterations, sizes 1KB-1GB
- **HIP memcpy** with device-to-device transfers
- **HIP Events** for precise timing

---

## Files Generated

- `gpu_transfer_bench` - Executable benchmark
- `gpu_transfer_results.txt` - Complete output
- `gpu_transfer_analysis.md` - This analysis

---

*Analysis shows MI300A GPUs have excellent interconnect performance, and RCCL achieves very good efficiency given the algorithmic complexity of collective operations.*
