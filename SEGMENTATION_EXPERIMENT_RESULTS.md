# Segmentation Experiment Results: 2 vs 3 Segments

## Experiment Design

Modified `segment_performance_bic.py` to try both 2 and 3 segments for each benchmark.
- **Selection Criteria**: Prefer 2 segments (simpler) unless 3 segments shows ≥10% BIC improvement
- **BIC**: Lower is better (Bayesian Information Criterion balances fit quality vs. complexity)

## Results Summary

| Benchmark        | Segments | BIC (2-seg) | BIC (3-seg) | Improvement | Reason                          |
|------------------|----------|-------------|-------------|-------------|---------------------------------|
| all_reduce       | **2**    | 185.97      | 188.66      | -1.4%       | 3-seg worse (preferred simplicity) |
| all_gather       | **2**    | 86.73       | 84.94       | 2.1%        | Below 10% threshold             |
| broadcast        | **2**    | 149.07      | 136.90      | 8.2%        | Below 10% threshold             |
| reduce           | **2**    | 158.86      | 148.35      | 6.6%        | Below 10% threshold             |
| reduce_scatter   | **3**    | 92.73       | 74.92       | **19.2%**   | ✓ Significant improvement       |
| scatter          | **3**    | 55.40       | 42.86       | **22.6%**   | ✓ Significant improvement       |
| gather           | **3**    | 80.23       | 62.17       | **22.5%**   | ✓ Significant improvement       |
| alltoall         | **2**    | 82.11       | 78.01       | 5.0%        | Below 10% threshold             |
| hypercube        | **2**    | 97.91       | 94.34       | 3.6%        | Below 10% threshold             |
| sendrecv         | **2**    | 104.16      | 101.11      | 2.9%        | Below 10% threshold             |
| all_reduce_bias  | ❌       | N/A         | N/A         | N/A         | Benchmark failed                |

## Key Findings

### 2 Segments Chosen (7 benchmarks)
Most benchmarks work well with 2 segments:
- **all_reduce, all_gather, broadcast, reduce, alltoall, hypercube, sendrecv**
- These show relatively smooth performance curves with one main transition point
- 3 segments either made it worse or didn't improve enough to justify added complexity

### 3 Segments Chosen (3 benchmarks)  
Three benchmarks benefit significantly from 3 segments:
- **reduce_scatter** (19.2% improvement): 128-256 → 512-262K → 524K-1G
- **scatter** (22.6% improvement): 128-1K → 2K-32K → 64K-1G
- **gather** (22.5% improvement): 128-256 → 512-32K → 64K-1G

These benchmarks show more complex performance behavior with multiple distinct regions, likely due to:
- Different communication patterns for small/medium/large messages
- Algorithm transitions in the RCCL implementation
- Memory hierarchy effects (L1/L2/L3/HBM transitions)

### Breakpoint Patterns

**2-Segment Breakpoints:**
- Most occur in the 16-256 MiB range
- Typical transition: latency-dominated → bandwidth-dominated

**3-Segment Breakpoints:**
- First break: Very small (256-512 bytes) - initialization overhead region
- Second break: Medium (32K-524K bytes) - algorithm/memory hierarchy transition
- Segments: latency/overhead → transition → bandwidth

## Algorithm Behavior

The 10% improvement threshold worked well:
- ✓ Prevents over-fitting with unnecessary segments
- ✓ Captures genuinely complex performance curves
- ✓ Maintains simplicity for most benchmarks
- ✓ Clear separation: 7 benchmarks with 2 segments, 3 with 3 segments

## Recommendations

1. **Keep the current implementation** - the 10% threshold is effective
2. **Use for production** - the algorithm makes sensible decisions automatically
3. **Document the patterns** - the 3-segment benchmarks (scatter, gather, reduce_scatter) show similar patterns and may share underlying RCCL implementation details

## Example Segmentations

### All Reduce (2 segments - simple case)
```
Segment 0: 8 B .. 128 MiB    → y = 45.39 + 5.81·size_MB (R²=0.9946)
Segment 1: 256 MiB .. 1 GiB  → y = 250.80 + 5.51·size_MB (R²=0.9989)
```

### Scatter (3 segments - complex case)
```
Segment 0: 128 B .. 1 KiB    → y = 36.83 - 1.59·log2(size) (R²=0.7001)
Segment 1: 2 KiB .. 32 KiB   → y = 33.79 - 0.16·log2(size) (R²=0.5663)
Segment 2: 64 KiB .. 1 GiB   → y = 21.75 + 2.84·size_MB (R²=1.0000)
```

## Date
November 8, 2025




