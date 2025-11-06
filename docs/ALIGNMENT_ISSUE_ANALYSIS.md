# RCCL-Tests Alignment Issue Analysis

## Executive Summary

**Issue:** Six RCCL benchmark implementations use an aggressive 16-byte alignment mask that causes small message sizes to be zeroed out when divided across multiple MPI ranks.

**Impact:** Benchmarks skip small message sizes (8-64 bytes depending on rank count), creating spurious zero-size entries and incomplete performance data.

**Root Cause:** The alignment formula `(count/nranks) & -(16/eltSize)` rounds down to 16-byte boundaries, which zeros out small per-rank message sizes.

---

## Affected Benchmarks

The following 6 benchmarks use the problematic alignment mask:

### 1. `all_gather_perf`
**File:** `src/all_gather.cu` line 13
```c
size_t base = (count/nranks) & -(16/eltSize);
```

### 2. `gather_perf`
**File:** `src/gather.cu` line 13
```c
*sendcount = (count/nranks) & -(16/eltSize);
```

### 3. `scatter_perf`
**File:** `src/scatter.cu` line 13
```c
*recvcount = (count/nranks) & -(16/eltSize);
```

### 4. `reduce_scatter_perf`
**File:** `src/reduce_scatter.cu` line 13
```c
size_t base = (count/nranks) & -(16/eltSize);
```

### 5. `alltoall_perf`
**File:** `src/alltoall.cu` line 13
```c
*paramcount = (count/nranks) & -(16/eltSize);
```

### 6. `hypercube_perf`
**File:** `src/hypercube.cu` line 14
```c
size_t base = (count/nranks) & -(16/eltSize);
```

---

## Safe Benchmarks

The following 5 benchmarks do NOT have this issue (they use `count` directly):

1. `all_reduce_perf` - Uses count directly
2. `all_reduce_bias_perf` - Uses count directly
3. `broadcast_perf` - Uses count directly
4. `reduce_perf` - Uses count directly
5. `sendrecv_perf` - Uses count directly

---

## Special Cases

### `alltoallv_perf`
**File:** `src/alltoallv.cu` line 15
Has an explicit check that sets counts to zero if `count < nranks*nranks/2`, which is a different mechanism but may have similar effects.

---

## Impact Analysis

### For float data type (4 bytes per element):

| Benchmark | 1 Rank | 2 Ranks | 4 Ranks | 8 Ranks |
|-----------|--------|---------|---------|---------|
| **First non-zero size** | 16 B | 32 B | 64 B | 128 B |
| **Skipped sizes** | 8 | 8, 16 | 8, 16, 32 | 8, 16, 32, 64 |

### Example: `alltoall_perf` with 8 ranks
- **Expected:** Test sizes 8, 16, 32, 64, 128, 256, ...
- **Actual:** Test sizes 0, 0, 0, 0, 128, 256, ...
- **Result:** Four zero-size entries, missing critical small-message data

---

## Technical Details

### The Alignment Mask

The mask `-(16/eltSize)` creates a bit pattern that rounds down to 16-byte boundaries:

For float (4 bytes):
- `16/eltSize = 16/4 = 4`
- `-(4) = 0xFFFFFFFC` (in two's complement)
- This mask clears the bottom 2 bits, rounding to multiples of 4 elements (16 bytes)

### Calculation Examples

**Size 8 bytes, 2 ranks, float (4 bytes):**
```
count = 8 / 4 = 2 elements
per_rank = 2 / 2 = 1 element
mask = -(16/4) = -4 = 0xFFFFFFFC
result = 1 & 0xFFFFFFFC = 0 elements → 0 bytes ✗
```

**Size 32 bytes, 2 ranks, float (4 bytes):**
```
count = 32 / 4 = 8 elements
per_rank = 8 / 2 = 4 elements
mask = -(16/4) = -4 = 0xFFFFFFFC
result = 4 & 0xFFFFFFFC = 4 elements → 16 bytes ✓
```

---

## Verification

### Test Results

**Command:**
```bash
python3 run_timing_sweep.py alltoall --datatype float --ranks 2 --mpi
```

**Expected first 3 sizes:** 8, 16, 32 bytes

**Actual benchmark output:**
```
size         count      type   ...
   0             0     float   ...  (should be 8)
   0             0     float   ...  (should be 16)
  32             4     float   ...  (correct)
```

This confirms the issue: sizes 8 and 16 are replaced with zero-size entries.

---

## Observed Behavior

### Symptoms
1. **Zero-size entries** appear in benchmark output
2. **Small message sizes are skipped** - the sweep jumps from 0 to larger sizes
3. **More ranks = more skipped sizes** - 8 ranks skip 8-64 byte range entirely
4. **Individual timing CSVs** show `size_bytes=0` for these entries
5. **Analysis scripts** fail to correlate timing data properly for these sizes

### Why This Wasn't Caught Earlier
- Single-rank runs only skip size 8 (often acceptable)
- Multi-rank runs are less common in quick tests
- Zero-size entries don't cause crashes, just incomplete data
- The alignment requirement may have been necessary for older hardware

---

## Recommendations

### Option 1: Remove or Relax Alignment (Preferred)
Change the mask to allow smaller alignments or remove it entirely:
```c
// Current (aggressive):
size_t base = (count/nranks) & -(16/eltSize);

// Option A: 4-byte alignment (sufficient for most types)
size_t base = (count/nranks) & -(4/eltSize);

// Option B: No alignment (test all sizes)
size_t base = (count/nranks);
```

### Option 2: Clamp to Minimum Size
Ensure the result is never zero:
```c
size_t base = (count/nranks) & -(16/eltSize);
if (base == 0 && count > 0) {
    base = 1;  // At least 1 element per rank
}
```

### Option 3: Document and Skip
If alignment is truly required, document it and adjust sweep parameters:
- Start sweeps at larger minimum sizes based on rank count
- Document the minimum size formula: `minBytes = 16 * nranks`

---

## Files Requiring Changes

If fixing the alignment issue:

1. `/work/lmeadows/rccl/rccl-tests/src/all_gather.cu` - line 13
2. `/work/lmeadows/rccl/rccl-tests/src/gather.cu` - line 13
3. `/work/lmeadows/rccl/rccl-tests/src/scatter.cu` - line 13
4. `/work/lmeadows/rccl/rccl-tests/src/reduce_scatter.cu` - line 13
5. `/work/lmeadows/rccl/rccl-tests/src/alltoall.cu` - line 13
6. `/work/lmeadows/rccl/rccl-tests/src/hypercube.cu` - line 14

---

## Related Issues

- **To-Do Item:** Debug zero size problem (occurs on all_gather and alltoall with >1 MPI rank)
- **User Report:** "Only one of the benchmark runs is saved" - led to discovery of file append issue
- **Analysis Impact:** Statistical analysis scripts see zero-size entries as valid data points

---

## Date
November 4, 2025

## Analyst
AI Assistant (Claude Sonnet 4.5)



