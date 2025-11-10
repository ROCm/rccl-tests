# Kernel Timeline Segmentation Experiment

## Goal

Modify `plot_kernel_timeline.py` to create one plot per performance segment instead of one large plot for all sizes, to improve creation time and rendering performance.

## Changes Made

### Modified `plot_kernel_timeline.py`

**Key changes:**
1. Added `load_segmentation()` function to load BIC segmentation JSON
2. Refactored main plotting logic into `plot_kernel_timeline_segment()` to create individual segment plots
3. Modified `plot_kernel_timeline()` to iterate over segments and create separate plots
4. Added timing and file size tracking for each segment
5. Updated `main()` to display summary statistics

**Filtering logic:**
- Timestamp events are filtered to only include sizes within each segment's range
- Kernels are filtered by time range (matching the filtered timestamp events with 10% padding)
- Each segment plot is independent with its own time axis starting from the segment's first event

## Test Results

### Benchmark: `all_reduce`
- **Run directory:** `/work/lmeadows/rccl/data/banff-ccs-aus-g01-14/run_all_reduce_20251108_194626`
- **Ranks:** 8
- **Iterations:** 100 per size
- **Segmentation:** 2 segments (from BIC analysis)

### Performance Metrics

#### Segment 0 (Small Messages)
- **Size range:** 8 - 262,144 bytes (16 sizes)
- **Creation time:** 439.91 seconds (~7.3 minutes)
- **File size:** 23.88 MB
- **Time span:** 3.12 seconds of kernel execution

#### Segment 1 (Large Messages)
- **Size range:** 524,288 - 1,073,741,824 bytes (12 sizes)
- **Creation time:** 382.61 seconds (~6.4 minutes)
- **File size:** 23.87 MB
- **Time span:** 3.12 seconds of kernel execution

#### Totals
- **Total creation time:** 822.52 seconds (~13.7 minutes)
- **Total file size:** 47.75 MB (sum of both segments)

## Analysis

### Benefits of Segmentation
1. **Logical organization:** Users can focus on one performance regime at a time
2. **Independent time axes:** Each segment's timeline starts at zero, making it easier to see relative timing
3. **Smaller individual files:** Each plot is ~24 MB instead of potentially 50+ MB for a combined plot
4. **Better browser performance:** Opening and rendering 24 MB is faster than 50+ MB

### Remaining Challenges
1. **Still slow to create:** Each segment takes 6-7 minutes to generate
2. **Still large files:** 24 MB per segment is still substantial
3. **Kernel count:** Each segment still contains thousands of kernel traces
   - Segment 0: 16 sizes × 100 iterations × 8 ranks = 12,800 benchmark runs
   - Each run may launch multiple NCCL kernels
   - Total: potentially 10,000-20,000 kernel traces per segment

### Why Files Are Still Large

The size is driven by the number of Plotly traces:
- Each kernel execution = 1 Scatter trace (with 2 points for start/end)
- Each Tstart/Tend marker = 1 VLine + 1 Scatter trace
- With thousands of kernels across 8 ranks, the HTML contains massive JSON data

## Recommendations for Further Optimization

### 1. **Sample kernels instead of showing all**
   - Show only every Nth iteration
   - Or show min/median/max execution patterns
   - Could reduce data by 10-100x

### 2. **Aggregate by size**
   - Show kernel patterns for each size, not each iteration
   - Use box plots or violin plots for iteration variability
   - Would reduce from 100 iterations to 1 summary per size

### 3. **Use canvas-based rendering**
   - Switch from Plotly to matplotlib or bokeh with WebGL
   - Canvas can handle many more elements than SVG/DOM
   - Trade-off: less interactive

### 4. **Create size-based drill-down**
   - Top-level view shows one representative iteration per size
   - Click to drill down to detailed view of that size
   - Lazy-load detailed data

### 5. **Use server-based visualization**
   - Datashader or similar for pre-aggregation
   - Stream/tile data on demand
   - Requires server infrastructure

## Conclusion

The segmentation approach **successfully separates the timeline by performance regime**, making the data more manageable and logically organized. However, **the fundamental issue remains the volume of kernel traces**.

Each segment still contains thousands of individual kernel executions, which creates:
- Long plot generation times (6-7 minutes per segment)
- Large file sizes (24 MB per segment)
- Potential browser performance issues when opening

**The script is functional and improved**, but for production use, consider implementing sampling or aggregation strategies to reduce the data volume by an order of magnitude.

## Files Modified

- `scripts/plot_kernel_timeline.py` - Refactored to create per-segment plots

## Output Files

Per-segment HTML files are created with naming pattern:
- `kernel_timeline_segment0.html`
- `kernel_timeline_segment1.html`
- etc.



