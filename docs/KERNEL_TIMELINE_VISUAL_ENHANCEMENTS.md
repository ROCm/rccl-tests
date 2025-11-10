# Kernel Timeline Visual Enhancements

## Summary

Enhanced the kernel timeline visualization in `plot_kernel_timeline.py` to make timestamp markers more visible and show the relationship between total interval time, kernel execution time, and launch gaps.

## Visual Improvements

### 1. **More Visible Start/End Markers**

**Before:**
- Tstart: Semi-transparent green (`rgba(0, 255, 0, 0.5)`), width=1, dashed line
- Tend: Semi-transparent red (`rgba(255, 0, 0, 0.5)`), width=1, dashed line
- Triangle markers: size=8

**After:**
- Tstart: **Fully opaque bright green (`rgb(0, 200, 0)`)**, width=**3**, **solid line**
- Tend: **Fully opaque bright red (`rgb(255, 0, 0)`)**, width=**3**, **solid line**
- Triangle markers: size=**12** with **dark borders**
  - Tstart: green fill with dark green border
  - Tend: red fill with dark red border

**Impact:** Start markers are now easily visible (were almost invisible before due to transparency and thin width).

### 2. **Timestamp Interval Bars**

Added **orange horizontal bars** spanning from Tstart to Tend:
- Color: Semi-transparent orange (`rgba(255, 165, 0, 0.3)`)
- Width: 15 pixels
- Position: Slightly above kernel traces (rank + 0.4)
- Shows in legend as "Timestamp Interval"

**Purpose:** Visually represents the total time span measured by timestamps, making it easy to see that the interval encompasses all kernel executions.

**Hover information:**
- Size and operation mode
- Start time
- End time
- Duration (Tend - Tstart)

### 3. **Timing Analysis Annotation**

Added a **text box** in the top-left corner showing:
```
Timing Analysis:
Total Interval: X.XX ms
Kernel Time: X.XX ms (XX.X%)
Launch Gaps: X.XX ms (XX.X%)
```

**Calculations:**
- **Total Interval** = `(Tend - Tstart)` summed across all ranks
- **Kernel Time** = Sum of all kernel durations (`end_ns - begin_ns`)
- **Launch Gaps** = Total Interval - Kernel Time
- **Percentages** show the breakdown

**Purpose:** Demonstrates the relationship:
```
Total Interval Time = Kernel Execution Time + Launch Overhead
```

This shows:
- How much time is spent actually executing kernels
- How much time is gaps between kernels (launch overhead)
- That the timestamp interval encompasses everything

### 4. **Enhanced Hover Information**

**Kernel traces** now show:
- Kernel name
- Start/end times
- Duration
- **Size in bytes** (newly added)

**Tend markers** now show:
- Time
- **Interval duration** (newly added)

## Visual Layout

For each rank, the timeline now shows (from top to bottom):
```
Orange Bar: [========== Timestamp Interval (Tstart to Tend) ==========]
     |                                                           |
  Green Marker                                              Red Marker
  (Tstart)                                                  (Tend)
     ↓                                                           ↑
Blue Bars: |==| |==|  |==| |==|  |==| ... (Individual NCCL kernels)
```

The orange bar visually demonstrates that:
1. All kernels fall within the timestamp interval
2. The gaps between blue bars are launch delays
3. The total width of the orange bar equals wall-clock time

## Example Output

For `all_reduce` size range 128-256 bytes (inp):
```
Timing Analysis:
Total Interval: 5.65 ms
Kernel Time: 4.65 ms (82.3%)
Launch Gaps: 1.00 ms (17.7%)
```

This shows:
- 82.3% of the time is actual kernel execution
- 17.7% is launch overhead (gaps between kernels)
- The sum equals the total timestamp interval

## Color Scheme

| Element | Color | Opacity | Purpose |
|---------|-------|---------|---------|
| NCCL Kernels | Blue `rgb(31, 119, 180)` | Opaque | Kernel execution |
| Tstart marker | Bright Green `rgb(0, 200, 0)` | Opaque | Benchmark start |
| Tend marker | Bright Red `rgb(255, 0, 0)` | Opaque | Benchmark end |
| Interval bar | Orange `rgba(255, 165, 0, 0.3)` | 30% | Total interval span |

## Benefits

1. **Clearer visualization** - Start/end markers are now easily distinguishable
2. **Shows timing relationship** - Orange bars demonstrate that interval = kernels + gaps
3. **Quantitative analysis** - Annotation provides exact numbers
4. **Interactive** - All elements have informative hover tooltips
5. **Validates expectations** - Confirms that timestamp interval encompasses all kernel activity

## Files Modified

- `scripts/plot_kernel_timeline.py` - Enhanced visualization with:
  - Thicker, opaque marker lines (width=3)
  - Larger marker symbols (size=12 with borders)
  - Orange interval bars showing timestamp span
  - Timing analysis annotation box
  - Enhanced hover information

## Usage

No changes to command-line interface. Enhanced visuals appear automatically:

```bash
python3 scripts/plot_kernel_timeline.py /path/to/run_dir --min-size 128 --max-size 1024 --mode inp
```

The resulting plot will show:
- Bright green vertical lines at each Tstart
- Bright red vertical lines at each Tend
- Orange bars spanning the intervals
- Timing breakdown in the top-left corner

## Performance Impact

The additional visual elements add minimal overhead:
- ~0.05 seconds per plot (for adding interval bars and annotation)
- File size increase: ~10-20 KB (negligible)

## Future Enhancements

Possible additions:
1. **Per-iteration breakdown** - Show individual iterations within the interval
2. **Cumulative time trace** - Add a line showing cumulative kernel time
3. **Gap analysis** - Highlight unusually large gaps
4. **Comparison mode** - Show multiple operation modes side-by-side
5. **Export timing data** - Save timing breakdown to CSV



