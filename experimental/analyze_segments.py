#!/usr/bin/env python3
"""
RCCL Benchmark Segmentation Analysis

Partitions message sizes into latency-bound and bandwidth-bound segments.

Approach:
1. Fit the alpha-beta model: time = alpha + beta * size
   - alpha = latency (fixed overhead)
   - beta = 1/bandwidth (transfer time per byte)

2. Find the crossover point where:
   - latency contribution = bandwidth contribution
   - i.e., alpha = beta * size
   - crossover_size = alpha / beta

3. Validate by checking bandwidth efficiency:
   - Latency-bound: bandwidth << peak (overhead dominates)
   - Bandwidth-bound: bandwidth approaches peak (transfer dominates)
"""

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple

import polars as pl
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


def load_benchmark_stats(results_dir: str) -> Dict[str, pl.DataFrame]:
    """Load benchmark statistics from all benchmarks."""
    benchmarks = {}
    
    for bench_dir in sorted(glob.glob(os.path.join(results_dir, "*"))):
        if not os.path.isdir(bench_dir):
            continue
        
        stats_file = os.path.join(bench_dir, "analysis", "benchmark_statistics.csv")
        if os.path.exists(stats_file):
            name = os.path.basename(bench_dir)
            benchmarks[name] = pl.read_csv(stats_file)
    
    return benchmarks


def alpha_beta_model(size: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    Alpha-beta latency model: time = alpha + beta * size
    
    alpha: fixed latency overhead (microseconds)
    beta: inverse bandwidth (microseconds per byte)
    """
    return alpha + beta * size


def fit_alpha_beta(sizes: np.ndarray, times: np.ndarray) -> Tuple[float, float]:
    """Fit alpha-beta model to data."""
    try:
        popt, _ = optimize.curve_fit(
            alpha_beta_model, 
            sizes, 
            times,
            p0=[10.0, 1e-6],  # Initial guess: 10μs latency, 1 GB/s
            bounds=([0, 0], [1e6, 1e-3]),  # Reasonable bounds
            maxfev=10000
        )
        return popt[0], popt[1]
    except Exception as e:
        print(f"  Fit failed: {e}")
        return None, None


def find_crossover_point(alpha: float, beta: float) -> float:
    """
    Find size where latency = transfer time.
    alpha = beta * size  =>  size = alpha / beta
    """
    if beta > 0:
        return alpha / beta
    return float('inf')


def find_bandwidth_knee(sizes: np.ndarray, bandwidths: np.ndarray, threshold: float = 0.5) -> float:
    """
    Find size where bandwidth reaches threshold fraction of peak.
    This is the "knee" of the bandwidth curve.
    """
    peak_bw = np.max(bandwidths)
    target_bw = threshold * peak_bw
    
    # Find first size where bandwidth exceeds threshold
    for i, bw in enumerate(bandwidths):
        if bw >= target_bw:
            return sizes[i]
    
    return sizes[-1]


def analyze_segments(
    df: pl.DataFrame,
    bench_name: str,
) -> Dict:
    """Analyze a single benchmark and find segment boundaries."""
    
    # Aggregate across ranks (use mean)
    agg = df.group_by(["size", "place"]).agg([
        pl.col("wall_time_us").mean().alias("time_us"),
        pl.col("wall_time_us").std().alias("time_std"),
    ]).sort("size")
    
    # Use out-of-place data (typically more consistent)
    oop = agg.filter(pl.col("place") == "oop")
    
    if len(oop) < 5:
        return None
    
    sizes = oop["size"].to_numpy().astype(float)
    times = oop["time_us"].to_numpy()
    
    # Compute bandwidth (GB/s)
    bandwidths = sizes / times / 1000.0
    
    # Fit alpha-beta model
    alpha, beta = fit_alpha_beta(sizes, times)
    
    if alpha is None:
        return None
    
    # Compute crossover point
    crossover_size = find_crossover_point(alpha, beta)
    
    # Compute bandwidth at crossover
    crossover_idx = np.searchsorted(sizes, crossover_size)
    if crossover_idx < len(bandwidths):
        crossover_bw = bandwidths[min(crossover_idx, len(bandwidths)-1)]
    else:
        crossover_bw = bandwidths[-1]
    
    # Find knee point (50% of peak bandwidth)
    knee_size = find_bandwidth_knee(sizes, bandwidths, threshold=0.5)
    
    # Find 80% of peak (near-saturation point)
    saturation_size = find_bandwidth_knee(sizes, bandwidths, threshold=0.8)
    
    # Model-predicted bandwidth
    model_bw = 1.0 / beta / 1000.0 if beta > 0 else 0  # GB/s
    
    peak_bw = np.max(bandwidths)
    
    return {
        "benchmark": bench_name,
        "alpha_us": alpha,
        "beta_us_per_byte": beta,
        "model_bandwidth_gbps": model_bw,
        "peak_bandwidth_gbps": peak_bw,
        "crossover_size": crossover_size,
        "crossover_bw_gbps": crossover_bw,
        "knee_50pct_size": knee_size,
        "saturation_80pct_size": saturation_size,
        "sizes": sizes,
        "times": times,
        "bandwidths": bandwidths,
    }


def create_segmentation_plot(results: List[Dict], output_dir: str):
    """Create a visualization of the segmentation for all benchmarks."""
    
    n_benchmarks = len(results)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, r in enumerate(results):
        if i >= len(axes):
            break
        
        ax = axes[i]
        sizes = r["sizes"]
        bandwidths = r["bandwidths"]
        times = r["times"]
        
        # Plot bandwidth vs size
        ax.semilogx(sizes, bandwidths, 'b-o', markersize=3, label='Measured')
        
        # Mark crossover point
        crossover = r["crossover_size"]
        if crossover < sizes[-1]:
            ax.axvline(crossover, color='red', linestyle='--', linewidth=2, 
                      label=f'Crossover: {crossover/1024:.0f} KB')
        
        # Mark knee point
        knee = r["knee_50pct_size"]
        ax.axvline(knee, color='green', linestyle=':', linewidth=2,
                  label=f'50% Peak: {knee/1024:.0f} KB')
        
        # Shade regions
        ax.axvspan(sizes[0], min(crossover, sizes[-1]), alpha=0.2, color='orange', 
                  label='Latency-bound')
        if crossover < sizes[-1]:
            ax.axvspan(crossover, sizes[-1], alpha=0.2, color='blue',
                      label='Bandwidth-bound')
        
        ax.set_xlabel('Size (bytes)')
        ax.set_ylabel('Bandwidth (GB/s)')
        ax.set_title(f'{r["benchmark"]}\nα={r["alpha_us"]:.1f}μs, peak={r["peak_bandwidth_gbps"]:.1f} GB/s')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, r["peak_bandwidth_gbps"] * 1.1)
    
    # Hide unused subplots
    for i in range(len(results), len(axes)):
        axes[i].set_visible(False)
    
    # Add legend to first plot
    axes[0].legend(loc='lower right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "segmentation_overview.png"), dpi=150)
    plt.savefig(os.path.join(output_dir, "segmentation_overview.pdf"))
    print(f"Saved: segmentation_overview.png/pdf")
    plt.close()


def create_summary_plot(results: List[Dict], output_dir: str):
    """Create a summary comparison across benchmarks."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    names = [r["benchmark"] for r in results]
    crossovers = [r["crossover_size"] / 1024 for r in results]  # KB
    alphas = [r["alpha_us"] for r in results]
    peaks = [r["peak_bandwidth_gbps"] for r in results]
    
    # Sort by crossover size
    sorted_idx = np.argsort(crossovers)
    names = [names[i] for i in sorted_idx]
    crossovers = [crossovers[i] for i in sorted_idx]
    alphas = [alphas[i] for i in sorted_idx]
    peaks = [peaks[i] for i in sorted_idx]
    
    x = np.arange(len(names))
    
    # Plot 1: Crossover sizes
    ax = axes[0]
    bars = ax.bar(x, crossovers)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Crossover Size (KB)')
    ax.set_title('Latency→Bandwidth Crossover Point')
    ax.set_yscale('log')
    
    # Add value labels
    for bar, val in zip(bars, crossovers):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
               f'{val:.0f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Latency (alpha)
    ax = axes[1]
    bars = ax.bar(x, alphas, color='orange')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Latency α (μs)')
    ax.set_title('Fixed Overhead (Latency)')
    
    for bar, val in zip(bars, alphas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
               f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Peak bandwidth
    ax = axes[2]
    bars = ax.bar(x, peaks, color='green')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Peak Bandwidth (GB/s)')
    ax.set_title('Peak Bandwidth')
    
    for bar, val in zip(bars, peaks):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
               f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "segmentation_summary.png"), dpi=150)
    print(f"Saved: segmentation_summary.png")
    plt.close()


def print_segmentation_table(results: List[Dict]):
    """Print a summary table of segmentation results."""
    
    print("\n" + "="*90)
    print("LATENCY-BOUND vs BANDWIDTH-BOUND SEGMENTATION")
    print("="*90)
    print("\nModel: time = α + β × size")
    print("  α (alpha) = fixed latency overhead")
    print("  β (beta)  = inverse bandwidth (time per byte)")
    print("  Crossover = size where α = β × size (latency = transfer time)")
    print()
    
    print(f"{'Benchmark':<15} {'α (μs)':<10} {'Peak BW':<12} {'Crossover':<15} {'Segment Boundary':}")
    print(f"{'':15} {'':10} {'(GB/s)':<12} {'(bytes)':<15} {'Recommendation':}")
    print("-" * 90)
    
    # Sort by crossover size
    sorted_results = sorted(results, key=lambda x: x["crossover_size"])
    
    for r in sorted_results:
        crossover = r["crossover_size"]
        
        # Format crossover nicely
        if crossover < 1024:
            crossover_str = f"{crossover:.0f} B"
        elif crossover < 1024 * 1024:
            crossover_str = f"{crossover/1024:.1f} KB"
        elif crossover < 1024 * 1024 * 1024:
            crossover_str = f"{crossover/1024/1024:.2f} MB"
        else:
            crossover_str = f"{crossover/1024/1024/1024:.2f} GB"
        
        # Recommendation
        if crossover < 64 * 1024:
            recommendation = "< 64 KB: latency | >= 64 KB: bandwidth"
        elif crossover < 256 * 1024:
            recommendation = "< 256 KB: latency | >= 256 KB: bandwidth"
        elif crossover < 1024 * 1024:
            recommendation = "< 1 MB: latency | >= 1 MB: bandwidth"
        else:
            recommendation = f"< {crossover_str}: latency | >= {crossover_str}: bandwidth"
        
        print(f"{r['benchmark']:<15} {r['alpha_us']:<10.1f} {r['peak_bandwidth_gbps']:<12.2f} {crossover_str:<15} {recommendation}")
    
    print()
    
    # Compute overall recommendation
    crossovers = [r["crossover_size"] for r in results]
    median_crossover = np.median(crossovers)
    
    if median_crossover < 1024:
        boundary_str = f"{median_crossover:.0f} B"
    elif median_crossover < 1024 * 1024:
        boundary_str = f"{median_crossover/1024:.0f} KB"
    else:
        boundary_str = f"{median_crossover/1024/1024:.1f} MB"
    
    print("="*90)
    print(f"RECOMMENDED OVERALL BOUNDARY: {boundary_str}")
    print(f"  - Sizes < {boundary_str}: LATENCY-BOUND (optimize for low overhead)")
    print(f"  - Sizes >= {boundary_str}: BANDWIDTH-BOUND (optimize for throughput)")
    print("="*90)
    
    return median_crossover


def save_results(results: List[Dict], median_crossover: float, output_dir: str):
    """Save segmentation results to files."""
    
    # Create summary dataframe
    rows = []
    for r in results:
        rows.append({
            "benchmark": r["benchmark"],
            "alpha_us": r["alpha_us"],
            "beta_us_per_byte": r["beta_us_per_byte"],
            "model_bandwidth_gbps": r["model_bandwidth_gbps"],
            "peak_bandwidth_gbps": r["peak_bandwidth_gbps"],
            "crossover_bytes": r["crossover_size"],
            "crossover_kb": r["crossover_size"] / 1024,
            "knee_50pct_bytes": r["knee_50pct_size"],
            "saturation_80pct_bytes": r["saturation_80pct_size"],
        })
    
    df = pl.DataFrame(rows)
    df.write_csv(os.path.join(output_dir, "segmentation_results.csv"))
    print(f"Saved: segmentation_results.csv")
    
    # Save boundary recommendation
    with open(os.path.join(output_dir, "segmentation_boundary.txt"), "w") as f:
        f.write(f"Recommended Segment Boundary: {median_crossover:.0f} bytes\n")
        f.write(f"  = {median_crossover/1024:.1f} KB\n")
        f.write(f"  = {median_crossover/1024/1024:.3f} MB\n")
        f.write(f"\n")
        f.write(f"Latency-bound: size < {median_crossover:.0f} bytes\n")
        f.write(f"Bandwidth-bound: size >= {median_crossover:.0f} bytes\n")
    print(f"Saved: segmentation_boundary.txt")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze RCCL benchmarks to find latency/bandwidth segments"
    )
    parser.add_argument(
        "--results-dir", "-r",
        required=True,
        help="Directory containing benchmark results"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory (default: results-dir)"
    )
    
    args = parser.parse_args()
    
    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading benchmark data...")
    benchmarks = load_benchmark_stats(args.results_dir)
    print(f"  Loaded {len(benchmarks)} benchmarks")
    
    print("\nAnalyzing segments...")
    results = []
    for name, df in benchmarks.items():
        print(f"  {name}...", end=" ")
        r = analyze_segments(df, name)
        if r:
            results.append(r)
            print(f"crossover at {r['crossover_size']/1024:.1f} KB")
        else:
            print("failed")
    
    if not results:
        print("No results to analyze!")
        return
    
    # Print summary table
    median_crossover = print_segmentation_table(results)
    
    # Create plots
    print("\nGenerating plots...")
    create_segmentation_plot(results, output_dir)
    create_summary_plot(results, output_dir)
    
    # Save results
    print("\nSaving results...")
    save_results(results, median_crossover, output_dir)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

