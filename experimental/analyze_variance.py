#!/usr/bin/env python3
"""
RCCL Benchmark Kernel Variance Analysis

Analyzes the variance of per-kernel execution times within each benchmark region.
Creates charts showing:
1. Bar chart of kernel time variance per message size
2. Line charts with error bars showing kernel timing variability
"""

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import polars as pl
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RegionKernelStats:
    """Statistics for kernels within a benchmark region."""
    rank: int
    size: int
    place: str  # 'oop' or 'inp'
    test: str
    wall_time_us: float
    kernel_times_ns: List[int]
    
    @property
    def num_kernels(self) -> int:
        return len(self.kernel_times_ns)
    
    @property
    def mean_kernel_ns(self) -> float:
        return np.mean(self.kernel_times_ns) if self.kernel_times_ns else 0
    
    @property
    def std_kernel_ns(self) -> float:
        return np.std(self.kernel_times_ns) if len(self.kernel_times_ns) > 1 else 0
    
    @property
    def var_kernel_ns(self) -> float:
        return np.var(self.kernel_times_ns) if len(self.kernel_times_ns) > 1 else 0
    
    @property
    def min_kernel_ns(self) -> float:
        return min(self.kernel_times_ns) if self.kernel_times_ns else 0
    
    @property
    def max_kernel_ns(self) -> float:
        return max(self.kernel_times_ns) if self.kernel_times_ns else 0
    
    @property
    def total_kernel_ns(self) -> float:
        return sum(self.kernel_times_ns)


def load_data(timestamps_pattern: str, parquet_dir: str) -> Tuple[Dict, Dict, Dict]:
    """Load timestamp CSVs and parquet data."""
    timestamps = {}
    for path in sorted(glob.glob(timestamps_pattern)):
        df = pl.read_csv(path)
        rank = int(Path(path).stem.split('_')[1])
        timestamps[rank] = df
    
    parquet_data = {}
    for pid_dir in sorted(glob.glob(os.path.join(parquet_dir, "*_parquet"))):
        pid = int(Path(pid_dir).name.replace("_parquet", ""))
        parquet_data[pid] = {}
        
        gpu_path = os.path.join(pid_dir, "timeline", "gpu_activity.parquet")
        if os.path.exists(gpu_path):
            parquet_data[pid]["gpu_activity"] = pl.read_parquet(gpu_path)
    
    pids = sorted(parquet_data.keys())
    pid_to_rank = {pid: i for i, pid in enumerate(pids)}
    
    return timestamps, parquet_data, pid_to_rank


def extract_region_kernel_stats(
    timestamps: Dict[int, pl.DataFrame],
    parquet_data: Dict[int, Dict[str, pl.DataFrame]],
    pid_to_rank: Dict[int, int],
) -> List[RegionKernelStats]:
    """Extract kernel statistics for each benchmark region."""
    
    results = []
    
    for rank, ts_df in timestamps.items():
        # Find PID for this rank
        pid = None
        for p, r in pid_to_rank.items():
            if r == rank:
                pid = p
                break
        
        if pid is None or pid not in parquet_data:
            continue
        
        gpu_activity = parquet_data[pid].get("gpu_activity", pl.DataFrame())
        if len(gpu_activity) == 0:
            continue
        
        # Pair up START/STOP events
        starts = ts_df.filter(pl.col("where") == "START")
        stops = ts_df.filter(pl.col("where") == "STOP")
        
        for i in range(len(starts)):
            start_row = starts.row(i, named=True)
            stop_row = stops.row(i, named=True)
            
            start_ts = start_row["ts"]
            stop_ts = stop_row["ts"]
            
            # Find kernels in this region
            in_region = gpu_activity.filter(
                (pl.col("timestamp_ns") >= start_ts) &
                (pl.col("timestamp_ns") + pl.col("duration_ns") <= stop_ts)
            )
            
            kernel_times = in_region["duration_ns"].to_list()
            
            stats = RegionKernelStats(
                rank=rank,
                size=start_row["size"],
                place=start_row["place"],
                test=start_row["test"],
                wall_time_us=(stop_ts - start_ts) / 1000.0,
                kernel_times_ns=kernel_times,
            )
            results.append(stats)
    
    return results


def create_variance_bar_chart(stats: List[RegionKernelStats], output_dir: str, test_name: str):
    """Create a bar chart showing kernel time variance for each size."""
    
    # Filter for specific test
    filtered = [s for s in stats if s.test == test_name]
    
    if not filtered:
        print(f"No data for test: {test_name}")
        return
    
    # Group by size and place, compute variance statistics
    size_data = {}
    for s in filtered:
        key = (s.size, s.place)
        if key not in size_data:
            size_data[key] = {"vars": [], "stds": [], "means": [], "cvs": []}
        
        size_data[key]["vars"].append(s.var_kernel_ns)
        size_data[key]["stds"].append(s.std_kernel_ns)
        size_data[key]["means"].append(s.mean_kernel_ns)
        if s.mean_kernel_ns > 0:
            size_data[key]["cvs"].append(s.std_kernel_ns / s.mean_kernel_ns * 100)
    
    # Aggregate across ranks
    sizes_oop = sorted(set(k[0] for k in size_data.keys() if k[1] == "oop"))
    sizes_inp = sorted(set(k[0] for k in size_data.keys() if k[1] == "inp"))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Variance (oop vs inp)
    ax = axes[0, 0]
    width = 0.35
    x = np.arange(len(sizes_oop))
    
    vars_oop = [np.mean(size_data.get((s, "oop"), {"vars": [0]})["vars"]) for s in sizes_oop]
    vars_inp = [np.mean(size_data.get((s, "inp"), {"vars": [0]})["vars"]) for s in sizes_inp]
    
    # Convert to microseconds squared for readability
    vars_oop_us2 = [v / 1e6 for v in vars_oop]
    vars_inp_us2 = [v / 1e6 for v in vars_inp]
    
    ax.bar(x - width/2, vars_oop_us2, width, label='Out-of-place', alpha=0.8)
    ax.bar(x + width/2, vars_inp_us2, width, label='In-place', alpha=0.8)
    ax.set_xlabel('Message Size (bytes)')
    ax.set_ylabel('Kernel Time Variance (μs²)')
    ax.set_title(f'{test_name} - Kernel Time Variance by Size')
    ax.set_xticks(x[::max(1, len(x)//10)])
    ax.set_xticklabels([f'{s:,}' for s in sizes_oop[::max(1, len(sizes_oop)//10)]], rotation=45, ha='right')
    ax.legend()
    ax.set_yscale('log')
    
    # Plot 2: Standard Deviation
    ax = axes[0, 1]
    stds_oop = [np.mean(size_data.get((s, "oop"), {"stds": [0]})["stds"]) / 1000 for s in sizes_oop]
    stds_inp = [np.mean(size_data.get((s, "inp"), {"stds": [0]})["stds"]) / 1000 for s in sizes_inp]
    
    ax.bar(x - width/2, stds_oop, width, label='Out-of-place', alpha=0.8)
    ax.bar(x + width/2, stds_inp, width, label='In-place', alpha=0.8)
    ax.set_xlabel('Message Size (bytes)')
    ax.set_ylabel('Kernel Time Std Dev (μs)')
    ax.set_title(f'{test_name} - Kernel Time Standard Deviation')
    ax.set_xticks(x[::max(1, len(x)//10)])
    ax.set_xticklabels([f'{s:,}' for s in sizes_oop[::max(1, len(sizes_oop)//10)]], rotation=45, ha='right')
    ax.legend()
    ax.set_yscale('log')
    
    # Plot 3: Coefficient of Variation
    ax = axes[1, 0]
    cvs_oop = [np.mean(size_data.get((s, "oop"), {"cvs": [0]})["cvs"]) for s in sizes_oop]
    cvs_inp = [np.mean(size_data.get((s, "inp"), {"cvs": [0]})["cvs"]) for s in sizes_inp]
    
    ax.bar(x - width/2, cvs_oop, width, label='Out-of-place', alpha=0.8)
    ax.bar(x + width/2, cvs_inp, width, label='In-place', alpha=0.8)
    ax.set_xlabel('Message Size (bytes)')
    ax.set_ylabel('Coefficient of Variation (%)')
    ax.set_title(f'{test_name} - Kernel Time CV (Std/Mean × 100)')
    ax.set_xticks(x[::max(1, len(x)//10)])
    ax.set_xticklabels([f'{s:,}' for s in sizes_oop[::max(1, len(sizes_oop)//10)]], rotation=45, ha='right')
    ax.legend()
    
    # Plot 4: Mean kernel time
    ax = axes[1, 1]
    means_oop = [np.mean(size_data.get((s, "oop"), {"means": [0]})["means"]) / 1000 for s in sizes_oop]
    means_inp = [np.mean(size_data.get((s, "inp"), {"means": [0]})["means"]) / 1000 for s in sizes_inp]
    
    ax.bar(x - width/2, means_oop, width, label='Out-of-place', alpha=0.8)
    ax.bar(x + width/2, means_inp, width, label='In-place', alpha=0.8)
    ax.set_xlabel('Message Size (bytes)')
    ax.set_ylabel('Mean Kernel Time (μs)')
    ax.set_title(f'{test_name} - Mean Kernel Time by Size')
    ax.set_xticks(x[::max(1, len(x)//10)])
    ax.set_xticklabels([f'{s:,}' for s in sizes_oop[::max(1, len(sizes_oop)//10)]], rotation=45, ha='right')
    ax.legend()
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{test_name.lower()}_variance_bars.png"), dpi=150)
    plt.savefig(os.path.join(output_dir, f"{test_name.lower()}_variance_bars.pdf"))
    print(f"Saved: {test_name.lower()}_variance_bars.png/pdf")
    plt.close()


def create_time_with_errorbars(stats: List[RegionKernelStats], output_dir: str, test_name: str):
    """Create line charts with X=size, Y=time, with error bars from kernel variance."""
    
    filtered = [s for s in stats if s.test == test_name]
    
    if not filtered:
        print(f"No data for test: {test_name}")
        return
    
    # Aggregate by size and place
    size_data = {}
    for s in filtered:
        key = (s.size, s.place)
        if key not in size_data:
            size_data[key] = {
                "wall_times": [],
                "total_kernel_times": [],
                "mean_kernels": [],
                "std_kernels": [],
                "min_kernels": [],
                "max_kernels": [],
            }
        
        size_data[key]["wall_times"].append(s.wall_time_us)
        size_data[key]["total_kernel_times"].append(s.total_kernel_ns / 1000.0)
        size_data[key]["mean_kernels"].append(s.mean_kernel_ns / 1000.0)
        size_data[key]["std_kernels"].append(s.std_kernel_ns / 1000.0)
        size_data[key]["min_kernels"].append(s.min_kernel_ns / 1000.0)
        size_data[key]["max_kernels"].append(s.max_kernel_ns / 1000.0)
    
    sizes = sorted(set(k[0] for k in size_data.keys()))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Wall time with std error bars (across ranks)
    ax = axes[0, 0]
    for place, marker, color in [("oop", "o", "tab:blue"), ("inp", "s", "tab:orange")]:
        plot_sizes = []
        means = []
        stds = []
        for s in sizes:
            if (s, place) in size_data:
                plot_sizes.append(s)
                times = size_data[(s, place)]["wall_times"]
                means.append(np.mean(times))
                stds.append(np.std(times))
        
        label = "Out-of-place" if place == "oop" else "In-place"
        ax.errorbar(plot_sizes, means, yerr=stds, marker=marker, capsize=3, 
                   label=label, color=color, alpha=0.8, markersize=4)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Message Size (bytes)')
    ax.set_ylabel('Wall Time (μs)')
    ax.set_title(f'{test_name} - Wall Time (error bars: std across ranks)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Mean kernel time with kernel std as error bars
    ax = axes[0, 1]
    for place, marker, color in [("oop", "o", "tab:blue"), ("inp", "s", "tab:orange")]:
        plot_sizes = []
        means = []
        stds = []
        for s in sizes:
            if (s, place) in size_data:
                plot_sizes.append(s)
                # Average of mean kernel times across ranks
                mk = size_data[(s, place)]["mean_kernels"]
                sk = size_data[(s, place)]["std_kernels"]
                means.append(np.mean(mk))
                stds.append(np.mean(sk))  # Use avg std as error bar
        
        label = "Out-of-place" if place == "oop" else "In-place"
        ax.errorbar(plot_sizes, means, yerr=stds, marker=marker, capsize=3,
                   label=label, color=color, alpha=0.8, markersize=4)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Message Size (bytes)')
    ax.set_ylabel('Mean Kernel Time (μs)')
    ax.set_title(f'{test_name} - Mean Kernel Time (error bars: kernel std dev)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Mean kernel time with min/max as error bars
    ax = axes[1, 0]
    for place, marker, color in [("oop", "o", "tab:blue"), ("inp", "s", "tab:orange")]:
        plot_sizes = []
        means = []
        err_low = []
        err_high = []
        for s in sizes:
            if (s, place) in size_data:
                plot_sizes.append(s)
                mk = np.mean(size_data[(s, place)]["mean_kernels"])
                min_k = np.mean(size_data[(s, place)]["min_kernels"])
                max_k = np.mean(size_data[(s, place)]["max_kernels"])
                means.append(mk)
                err_low.append(mk - min_k)
                err_high.append(max_k - mk)
        
        label = "Out-of-place" if place == "oop" else "In-place"
        ax.errorbar(plot_sizes, means, yerr=[err_low, err_high], marker=marker, capsize=3,
                   label=label, color=color, alpha=0.8, markersize=4)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Message Size (bytes)')
    ax.set_ylabel('Mean Kernel Time (μs)')
    ax.set_title(f'{test_name} - Mean Kernel Time (error bars: min/max range)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Total GPU time with std across ranks
    ax = axes[1, 1]
    for place, marker, color in [("oop", "o", "tab:blue"), ("inp", "s", "tab:orange")]:
        plot_sizes = []
        means = []
        stds = []
        for s in sizes:
            if (s, place) in size_data:
                plot_sizes.append(s)
                tk = size_data[(s, place)]["total_kernel_times"]
                means.append(np.mean(tk))
                stds.append(np.std(tk))
        
        label = "Out-of-place" if place == "oop" else "In-place"
        ax.errorbar(plot_sizes, means, yerr=stds, marker=marker, capsize=3,
                   label=label, color=color, alpha=0.8, markersize=4)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Message Size (bytes)')
    ax.set_ylabel('Total GPU Time (μs)')
    ax.set_title(f'{test_name} - Total GPU Time (error bars: std across ranks)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{test_name.lower()}_time_errorbars.png"), dpi=150)
    plt.savefig(os.path.join(output_dir, f"{test_name.lower()}_time_errorbars.pdf"))
    print(f"Saved: {test_name.lower()}_time_errorbars.png/pdf")
    plt.close()


def create_per_rank_variance_chart(stats: List[RegionKernelStats], output_dir: str, test_name: str):
    """Create charts showing variance per rank."""
    
    filtered = [s for s in stats if s.test == test_name]
    
    if not filtered:
        return
    
    ranks = sorted(set(s.rank for s in filtered))
    sizes = sorted(set(s.size for s in filtered))
    
    # Just do out-of-place for clarity
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Heatmap of std dev by rank and size
    ax = axes[0]
    std_matrix = np.zeros((len(ranks), len(sizes)))
    
    for s in filtered:
        if s.place == "oop":
            ri = ranks.index(s.rank)
            si = sizes.index(s.size)
            std_matrix[ri, si] = s.std_kernel_ns / 1000.0  # Convert to μs
    
    im = ax.imshow(std_matrix, aspect='auto', cmap='YlOrRd')
    ax.set_xlabel('Size Index')
    ax.set_ylabel('Rank')
    ax.set_yticks(range(len(ranks)))
    ax.set_yticklabels([f'Rank {r}' for r in ranks])
    ax.set_title(f'{test_name} - Kernel Std Dev (μs) by Rank and Size (OOP)')
    plt.colorbar(im, ax=ax, label='Std Dev (μs)')
    
    # Line plot showing variance per rank for large sizes
    ax = axes[1]
    large_sizes = [s for s in sizes if s >= 1048576][:5]  # Top 5 large sizes
    
    for size in large_sizes:
        rank_stds = []
        for rank in ranks:
            matching = [s.std_kernel_ns / 1000 for s in filtered 
                       if s.rank == rank and s.size == size and s.place == "oop"]
            if matching:
                rank_stds.append(matching[0])
            else:
                rank_stds.append(0)
        
        ax.plot(ranks, rank_stds, 'o-', label=f'{size:,} bytes', markersize=6)
    
    ax.set_xlabel('Rank')
    ax.set_ylabel('Kernel Std Dev (μs)')
    ax.set_title(f'{test_name} - Kernel Variance by Rank (large sizes, OOP)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{test_name.lower()}_rank_variance.png"), dpi=150)
    print(f"Saved: {test_name.lower()}_rank_variance.png")
    plt.close()


def save_variance_stats(stats: List[RegionKernelStats], output_dir: str):
    """Save detailed variance statistics to CSV."""
    
    rows = []
    for s in stats:
        rows.append({
            "rank": s.rank,
            "size": s.size,
            "place": s.place,
            "test": s.test,
            "wall_time_us": s.wall_time_us,
            "num_kernels": s.num_kernels,
            "total_kernel_us": s.total_kernel_ns / 1000.0,
            "mean_kernel_us": s.mean_kernel_ns / 1000.0,
            "std_kernel_us": s.std_kernel_ns / 1000.0,
            "var_kernel_us2": s.var_kernel_ns / 1e6,
            "min_kernel_us": s.min_kernel_ns / 1000.0,
            "max_kernel_us": s.max_kernel_ns / 1000.0,
            "cv_percent": (s.std_kernel_ns / s.mean_kernel_ns * 100) if s.mean_kernel_ns > 0 else 0,
        })
    
    df = pl.DataFrame(rows)
    df.write_csv(os.path.join(output_dir, "kernel_variance_stats.csv"))
    df.write_parquet(os.path.join(output_dir, "kernel_variance_stats.parquet"))
    print(f"Saved: kernel_variance_stats.csv/parquet")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Analyze kernel timing variance in RCCL benchmarks"
    )
    parser.add_argument(
        "--timestamps", "-t",
        default="rank_*_timestamps.csv",
        help="Glob pattern for timestamp CSV files"
    )
    parser.add_argument(
        "--parquet-dir", "-p",
        required=True,
        help="Directory containing Parquet output"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="analysis_output",
        help="Directory for output"
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading data...")
    timestamps, parquet_data, pid_to_rank = load_data(args.timestamps, args.parquet_dir)
    print(f"  Loaded {len(timestamps)} ranks, {len(parquet_data)} processes")
    
    print("\nExtracting kernel statistics per region...")
    stats = extract_region_kernel_stats(timestamps, parquet_data, pid_to_rank)
    print(f"  Extracted stats for {len(stats)} regions")
    
    if not stats:
        print("No statistics collected!")
        return
    
    # Save detailed stats
    print("\nSaving variance statistics...")
    df = save_variance_stats(stats, args.output_dir)
    
    # Get unique test names
    test_names = list(set(s.test for s in stats))
    
    # Create plots for each test
    for test_name in test_names:
        print(f"\nGenerating plots for: {test_name}")
        create_variance_bar_chart(stats, args.output_dir, test_name)
        create_time_with_errorbars(stats, args.output_dir, test_name)
        create_per_rank_variance_chart(stats, args.output_dir, test_name)
    
    # Print summary
    print("\n" + "="*60)
    print("VARIANCE ANALYSIS SUMMARY")
    print("="*60)
    
    for test_name in test_names:
        filtered = [s for s in stats if s.test == test_name]
        sizes = sorted(set(s.size for s in filtered))
        
        print(f"\n{test_name}:")
        print(f"  Sizes: {len(sizes)} ({min(sizes):,} - {max(sizes):,} bytes)")
        
        # Large message stats
        large = [s for s in filtered if s.size >= 1048576]
        if large:
            avg_cv = np.mean([s.std_kernel_ns / s.mean_kernel_ns * 100 
                            for s in large if s.mean_kernel_ns > 0])
            print(f"  Avg CV (>=1MB): {avg_cv:.1f}%")
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

