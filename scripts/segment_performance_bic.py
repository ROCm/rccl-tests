#!/usr/bin/env python3
"""
Performance segmentation using Piecewise Linear Regression with BIC.

This algorithm finds the optimal 2 breakpoints to create exactly 3 segments,
using Bayesian Information Criterion (BIC) to balance fit quality and complexity.
"""

import argparse
import json
import os
import sys
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd


def load_timing_data(output_dir):
    """Load timing CSV files from output directory."""
    timing_files = []
    for filename in os.listdir(output_dir):
        if filename.endswith('_rank0.csv') or (filename.endswith('.csv') and '_rank' in filename):
            timing_files.append(os.path.join(output_dir, filename))
    
    if not timing_files:
        return None
    
    # Load and combine all rank files
    dfs = []
    for filepath in timing_files:
        df = pd.read_csv(filepath)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def load_benchmark_output(output_dir, benchmark_name):
    """Load benchmark output file."""
    output_file = os.path.join(output_dir, f'{benchmark_name}_benchmark_output.txt')
    if not os.path.exists(output_file):
        return None
    
    with open(output_file, 'r') as f:
        return f.read()


def parse_benchmark_output(content):
    """Parse benchmark output to extract wall-clock times."""
    import re
    
    # Pattern: size count type op root time_oop algbw busbw errors time_ip ...
    pattern = r'^\s*(\d+)\s+(\d+)\s+(\w+)\s+(\w+)\s+(-?\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)'
    
    results = []
    for line in content.split('\n'):
        match = re.match(pattern, line)
        if match:
            size_bytes = int(match.group(1))
            time_oop = float(match.group(6))
            time_ip_match = re.search(r'\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+', line[match.end():])
            time_ip = float(time_ip_match.group(1)) if time_ip_match else time_oop
            
            results.append({
                'size_bytes': size_bytes,
                'wall_time_oop_us': time_oop,
                'wall_time_ip_us': time_ip
            })
    
    return pd.DataFrame(results)


def fit_linear(x, y):
    """Fit linear model: y = a + b*x"""
    if len(x) < 2:
        return None
    
    try:
        # Convert to MB for numerical stability
        x_mb = np.array(x) / (1024 * 1024)
        y_arr = np.array(y)
        
        # Fit y = a + b*x_mb
        A = np.vstack([np.ones(len(x_mb)), x_mb]).T
        params, residuals, rank, s = np.linalg.lstsq(A, y_arr, rcond=None)
        a, b = params
        
        # Compute R²
        y_pred = a + b * x_mb
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'model': 'linear',
            'formula': f'y = {a:.2f} + {b:.2f}·size_MB',
            'params': {'a': a, 'b': b},
            'r2': r2,
            'rss': ss_res,
            'n': len(x)
        }
    except:
        return None


def fit_loglinear(x, y):
    """Fit log-linear model: y = a + b*log2(x)"""
    if len(x) < 2:
        return None
    
    try:
        x_arr = np.array(x)
        y_arr = np.array(y)
        
        # Filter out zero or negative sizes
        valid = x_arr > 0
        if not np.any(valid):
            return None
        
        x_arr = x_arr[valid]
        y_arr = y_arr[valid]
        
        # Fit y = a + b*log2(x)
        log_x = np.log2(x_arr)
        A = np.vstack([np.ones(len(log_x)), log_x]).T
        params, residuals, rank, s = np.linalg.lstsq(A, y_arr, rcond=None)
        a, b = params
        
        # Compute R²
        y_pred = a + b * log_x
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'model': 'log-linear',
            'formula': f'y = {a:.2f} + {b:.2f}·log2(size)',
            'params': {'a': a, 'b': b},
            'r2': r2,
            'rss': ss_res,
            'n': len(x_arr)
        }
    except:
        return None


def fit_segment(sizes, times):
    """Fit both linear and log-linear models, return best."""
    linear_fit = fit_linear(sizes, times)
    loglin_fit = fit_loglinear(sizes, times)
    
    # Choose model with better R²
    if linear_fit is None and loglin_fit is None:
        return None
    elif linear_fit is None:
        return loglin_fit
    elif loglin_fit is None:
        return linear_fit
    else:
        return linear_fit if linear_fit['r2'] >= loglin_fit['r2'] else loglin_fit


def compute_bic_for_breaks(sizes, times, breaks):
    """
    Compute BIC for given breakpoint configuration.
    
    BIC = n·log(RSS/n) + k·log(n)
    where:
        n = total number of data points
        k = number of parameters (2 per segment for linear/log-linear)
        RSS = residual sum of squares
    """
    n = len(sizes)
    n_segments = len(breaks) + 1
    k = 2 * n_segments  # 2 parameters (a, b) per segment
    
    total_rss = 0
    start_idx = 0
    
    for end_idx in breaks + [n]:
        seg_sizes = sizes[start_idx:end_idx]
        seg_times = times[start_idx:end_idx]
        
        if len(seg_sizes) < 2:
            return np.inf  # Invalid segmentation
        
        fit = fit_segment(seg_sizes, seg_times)
        if fit is None:
            return np.inf
        
        total_rss += fit['rss']
        start_idx = end_idx
    
    if total_rss <= 0:
        return np.inf
    
    # BIC formula
    bic = n * np.log(total_rss / n) + k * np.log(n)
    return bic


def find_optimal_breakpoints(sizes, times, n_segments=3):
    """
    Find optimal breakpoints for n_segments using exhaustive search with BIC.
    
    Returns:
        best_breaks: List of breakpoint indices
        segments: List of segment fit information
        bic: BIC value for best segmentation
    """
    n = len(sizes)
    n_breaks = n_segments - 1
    
    if n < 2 * n_segments + 1:
        raise ValueError(f"Not enough data points ({n}) for {n_segments} segments")
    
    print(f"Searching for optimal {n_breaks} breakpoints among {n} data points...")
    
    best_bic = np.inf
    best_breaks = None
    
    # Exhaustive search for 2 breakpoints
    # Ensure minimum 2 points per segment
    min_seg_size = 2
    
    for i in range(min_seg_size, n - 2*min_seg_size):
        for j in range(i + min_seg_size, n - min_seg_size):
            breaks = [i, j]
            bic = compute_bic_for_breaks(sizes, times, breaks)
            
            if bic < best_bic:
                best_bic = bic
                best_breaks = breaks
    
    if best_breaks is None:
        raise ValueError("Could not find valid breakpoints")
    
    print(f"Optimal breakpoints found at indices: {best_breaks}")
    print(f"BIC = {best_bic:.2f}")
    
    # Fit models to each segment
    segments = []
    start_idx = 0
    
    for seg_num, end_idx in enumerate(best_breaks + [n]):
        seg_sizes = sizes[start_idx:end_idx]
        seg_times = times[start_idx:end_idx]
        
        fit = fit_segment(seg_sizes, seg_times)
        if fit is not None:
            fit['segment'] = seg_num
            fit['size_range'] = (seg_sizes[0], seg_sizes[-1])
            fit['index_range'] = (start_idx, end_idx - 1)
            segments.append(fit)
        
        start_idx = end_idx
    
    return best_breaks, segments, best_bic


def segment_benchmark_data(output_dir, benchmark_name):
    """Main segmentation function."""
    
    print(f"Segmenting {benchmark_name} benchmark data from {output_dir}")
    print("Using: Piecewise Linear Regression with BIC (3 segments)")
    print("=" * 80)
    
    # Load timing data
    timing_df = load_timing_data(output_dir)
    
    # Load benchmark output
    benchmark_content = load_benchmark_output(output_dir, benchmark_name)
    if benchmark_content is None:
        print(f"Error: Could not load benchmark output")
        return None
    
    benchmark_df = parse_benchmark_output(benchmark_content)
    if benchmark_df.empty:
        print(f"Error: Could not parse benchmark output")
        return None
    
    print(f"Parsed {len(benchmark_df)} timing entries from benchmark output")
    
    # Prepare data for segmentation
    # Use out-of-place wall time as the target
    sizes = benchmark_df['size_bytes'].values
    times = benchmark_df['wall_time_oop_us'].values
    
    # Filter out zero sizes
    valid = sizes > 0
    sizes = sizes[valid]
    times = times[valid]
    
    if len(sizes) < 7:
        print(f"Error: Not enough data points ({len(sizes)}) for 3-segment analysis")
        return None
    
    print(f"Data points: {len(sizes)}")
    print(f"Size range: {sizes[0]} to {sizes[-1]} bytes")
    print()
    
    # Find optimal segmentation
    try:
        breakpoints, segments, bic = find_optimal_breakpoints(sizes, times, n_segments=3)
    except Exception as e:
        print(f"Error during segmentation: {e}")
        return None
    
    # Convert breakpoint indices to sizes
    break_sizes = [sizes[bp] for bp in breakpoints]
    
    # Create output structure
    result = {
        'benchmark': benchmark_name,
        'algorithm': 'piecewise_linear_bic',
        'n_segments': 3,
        'n_datapoints': int(len(sizes)),
        'bic': float(bic),
        'breakpoint_indices': [int(x) for x in breakpoints],
        'breakpoint_sizes': [int(x) for x in break_sizes],
        'segments': []
    }
    
    # Add segment information
    for seg in segments:
        result['segments'].append({
            'segment': int(seg['segment']),
            'size_range_bytes': [int(x) for x in seg['size_range']],
            'index_range': [int(x) for x in seg['index_range']],
            'n_points': int(seg['n']),
            'model': seg['model'],
            'formula': seg['formula'],
            'parameters': {k: float(v) for k, v in seg['params'].items()},
            'r_squared': float(seg['r2'])
        })
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Segment performance data using Piecewise Linear Regression with BIC'
    )
    parser.add_argument('run_dir',
                       help='Run directory containing benchmark data')
    
    args = parser.parse_args()
    
    # Extract benchmark name from directory
    import re
    dir_basename = os.path.basename(args.run_dir.rstrip('/'))
    # Pattern: run_{benchmark}_{timestamp}
    match = re.match(r'run_([a-z_]+)_\d{8}_\d{6}', dir_basename)
    if match:
        benchmark_name = match.group(1)
    else:
        print(f"Error: Could not extract benchmark name from directory: {dir_basename}")
        print(f"Expected format: run_<benchmark>_YYYYMMDD_HHMMSS")
        return 1
    
    # Run segmentation
    result = segment_benchmark_data(args.run_dir, benchmark_name)
    
    if result is None:
        return 1
    
    # Save results
    output_file = os.path.join(args.run_dir, f'{benchmark_name}_bic_segmentation.json')
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print()
    print("=" * 80)
    print("SEGMENTATION RESULTS")
    print("=" * 80)
    print(f"Benchmark: {result['benchmark']}")
    print(f"Algorithm: Piecewise Linear Regression with BIC")
    print(f"BIC: {result['bic']:.2f}")
    print(f"Breakpoints at sizes: {', '.join(str(s) for s in result['breakpoint_sizes'])} bytes")
    print()
    
    for seg in result['segments']:
        size_min, size_max = seg['size_range_bytes']
        print(f"Segment {seg['segment']}: {size_min}..{size_max} bytes")
        print(f"  Model: {seg['model']}")
        print(f"  Formula: {seg['formula']}")
        print(f"  R² = {seg['r_squared']:.4f}")
        print(f"  N = {seg['n_points']} points")
        print()
    
    print(f"Results saved to: {output_file}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

