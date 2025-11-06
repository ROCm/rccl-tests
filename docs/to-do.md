# To-Do

- [x] **[COMPLETED]** Fix alignment issue in 6 benchmarks - WORKAROUND IMPLEMENTED
  - **Root cause identified:** Aggressive 16-byte alignment mask `-(16/eltSize)` zeros out small per-rank message sizes
  - **Workaround:** Modified `run_timing_sweep.py` to automatically adjust minimum size based on benchmark and rank count
  - **Status:** Script now prevents zero-size outputs by starting sweeps at safe minimum (nranks × 16 bytes, rounded to power of 2)
  - **Verified:** Tested with alltoall (2 ranks → 32B min, 8 ranks → 128B min) and all_reduce (always 8B min)
  - **See:** `docs/ALIGNMENT_FIX_IMPLEMENTATION.md` for implementation details
  - **See:** `docs/ALIGNMENT_ISSUE_ANALYSIS.md` for root cause analysis
  - **Future work:** Consider fixing root cause in benchmark source files (all_gather.cu, gather.cu, scatter.cu, reduce_scatter.cu, alltoall.cu, hypercube.cu)
