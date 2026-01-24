#!/usr/bin/env python3
"""
Advanced RCCL Kernel Analysis Tool

This script provides detailed kernel-level analysis including:
- Kernel breakdown per collective operation
- Cross-rank kernel timing comparison
- RCCL protocol/algorithm inference from kernel patterns
"""

import argparse
import glob
import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import polars as pl
import matplotlib.pyplot as plt
import numpy as np


def load_all_data(timestamps_pattern: str, parquet_dir: str) -> Tuple[Dict, Dict, Dict]:
    """Load all timestamp and parquet data."""
    
    # Load timestamps
    timestamps = {}
    for path in sorted(glob.glob(timestamps_pattern)):
        df = pl.read_csv(path)
        rank = int(Path(path).stem.split('_')[1])
        timestamps[rank] = df
    
    # Load parquet data
    parquet_data = {}
    for pid_dir in sorted(glob.glob(os.path.join(parquet_dir, "*_parquet"))):
        pid = int(Path(pid_dir).name.replace("_parquet", ""))
        parquet_data[pid] = {}
        
        for subdir in ["timeline", "details", "reference"]:
            subdir_path = os.path.join(pid_dir, subdir)
            if os.path.isdir(subdir_path):
                for f in os.listdir(subdir_path):
                    if f.endswith(".parquet"):
                        name = f.replace(".parquet", "")
                        parquet_data[pid][name] = pl.read_parquet(os.path.join(subdir_path, f))
    
    # Create pid to rank mapping
    pids = sorted(parquet_data.keys())
    pid_to_rank = {pid: i for i, pid in enumerate(pids)}
    
    return timestamps, parquet_data, pid_to_rank


def analyze_kernel_patterns(parquet_data: Dict, pid_to_rank: Dict) -> pl.DataFrame:
    """Analyze kernel patterns across all ranks."""
    
    all_kernel_stats = []
    
    for pid, pdata in parquet_data.items():
        rank = pid_to_rank[pid]
        
        gpu_activity = pdata.get("gpu_activity", pl.DataFrame())
        kernel_info = pdata.get("kernel_info", pl.DataFrame())
        
        if len(gpu_activity) == 0 or len(kernel_info) == 0:
            continue
        
        # Join to get kernel names
        activity = gpu_activity.join(
            kernel_info.select(["kernel_id", "kernel_name", "truncated_name"]),
            on="kernel_id",
            how="left"
        )
        
        # Group by kernel name
        kernel_stats = activity.group_by("truncated_name").agg([
            pl.count().alias("count"),
            pl.col("duration_ns").sum().alias("total_duration_ns"),
            pl.col("duration_ns").mean().alias("avg_duration_ns"),
            pl.col("duration_ns").min().alias("min_duration_ns"),
            pl.col("duration_ns").max().alias("max_duration_ns"),
            pl.col("duration_ns").std().alias("std_duration_ns"),
        ]).with_columns([
            pl.lit(rank).alias("rank"),
            pl.lit(pid).alias("pid"),
        ])
        
        all_kernel_stats.append(kernel_stats)
    
    if all_kernel_stats:
        return pl.concat(all_kernel_stats)
    return pl.DataFrame()


def identify_rccl_kernels(kernel_stats: pl.DataFrame) -> Dict[str, List[str]]:
    """Identify RCCL-related kernels by name patterns."""
    
    categories = {
        "rccl_reduce": [],
        "rccl_allreduce": [],
        "rccl_broadcast": [],
        "rccl_allgather": [],
        "rccl_reduce_scatter": [],
        "rccl_send_recv": [],
        "rccl_other": [],
        "hip_memset": [],
        "hip_memcpy": [],
        "verification": [],
        "other": [],
    }
    
    kernel_names = kernel_stats["truncated_name"].unique().to_list()
    
    for name in kernel_names:
        if name is None:
            continue
        name_lower = name.lower()
        
        if "ncclkernel" in name_lower or "rcclkernel" in name_lower:
            if "allreduce" in name_lower:
                categories["rccl_allreduce"].append(name)
            elif "reduce_scatter" in name_lower or "reducescatter" in name_lower:
                categories["rccl_reduce_scatter"].append(name)
            elif "reduce" in name_lower:
                categories["rccl_reduce"].append(name)
            elif "broadcast" in name_lower:
                categories["rccl_broadcast"].append(name)
            elif "allgather" in name_lower:
                categories["rccl_allgather"].append(name)
            elif "send" in name_lower or "recv" in name_lower:
                categories["rccl_send_recv"].append(name)
            else:
                categories["rccl_other"].append(name)
        elif "memset" in name_lower:
            categories["hip_memset"].append(name)
        elif "memcpy" in name_lower or "copy" in name_lower:
            categories["hip_memcpy"].append(name)
        elif "verif" in name_lower or "check" in name_lower:
            categories["verification"].append(name)
        else:
            categories["other"].append(name)
    
    return categories


def create_kernel_breakdown_plot(kernel_stats: pl.DataFrame, output_dir: str):
    """Create a plot showing kernel time breakdown."""
    
    # Aggregate across all ranks
    agg_stats = kernel_stats.group_by("truncated_name").agg([
        pl.col("count").sum().alias("total_count"),
        pl.col("total_duration_ns").sum().alias("total_time_ns"),
    ]).sort("total_time_ns", descending=True)
    
    # Take top 15 kernels
    top_kernels = agg_stats.head(15)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # By time
    ax = axes[0]
    names = top_kernels["truncated_name"].to_list()
    times = top_kernels["total_time_ns"].to_numpy() / 1e6  # Convert to ms
    
    # Truncate long names
    names = [n[:40] + "..." if len(n) > 40 else n for n in names]
    
    y_pos = range(len(names))
    ax.barh(y_pos, times)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Total Time (ms)")
    ax.set_title("Top Kernels by Total Execution Time")
    
    # By count
    ax = axes[1]
    counts = top_kernels["total_count"].to_numpy()
    ax.barh(y_pos, counts)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Invocation Count")
    ax.set_title("Top Kernels by Invocation Count")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "kernel_breakdown.png"), dpi=150)
    print(f"Saved: kernel_breakdown.png")
    plt.close()


def create_cross_rank_comparison(kernel_stats: pl.DataFrame, output_dir: str):
    """Create a plot comparing kernel timing across ranks."""
    
    # Get RCCL kernels that appear on all ranks
    ranks = kernel_stats["rank"].unique().to_list()
    
    # Find kernels that contain "ncclKernel" or "rcclKernel"
    rccl_stats = kernel_stats.filter(
        pl.col("truncated_name").str.contains("(?i)ncclkernel|rcclkernel")
    )
    
    if len(rccl_stats) == 0:
        print("No RCCL kernels found for cross-rank comparison")
        return
    
    # Aggregate by kernel name and rank
    pivot_data = rccl_stats.group_by(["truncated_name", "rank"]).agg([
        pl.col("avg_duration_ns").mean().alias("avg_duration_ns"),
        pl.col("count").sum().alias("count"),
    ])
    
    # Get unique kernel names
    kernel_names = pivot_data["truncated_name"].unique().to_list()
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(kernel_names) * 0.4)))
    
    bar_width = 0.2
    positions = np.arange(len(kernel_names))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(ranks)))
    
    for i, rank in enumerate(sorted(ranks)):
        rank_data = pivot_data.filter(pl.col("rank") == rank)
        
        times = []
        for name in kernel_names:
            row = rank_data.filter(pl.col("truncated_name") == name)
            if len(row) > 0:
                times.append(row["avg_duration_ns"].item() / 1000.0)  # Convert to μs
            else:
                times.append(0)
        
        ax.barh(positions + i * bar_width, times, bar_width, 
                label=f"Rank {rank}", color=colors[i])
    
    # Truncate long names
    short_names = [n[:50] + "..." if len(n) > 50 else n for n in kernel_names]
    
    ax.set_yticks(positions + bar_width * (len(ranks) - 1) / 2)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Average Duration (μs)")
    ax.set_title("RCCL Kernel Timing Comparison Across Ranks")
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cross_rank_kernel_comparison.png"), dpi=150)
    print(f"Saved: cross_rank_kernel_comparison.png")
    plt.close()


def create_kernel_timeline_heatmap(
    timestamps: Dict,
    parquet_data: Dict,
    pid_to_rank: Dict,
    size: int,
    output_dir: str,
):
    """Create a heatmap showing kernel activity across ranks for a specific size."""
    
    # Find the benchmark region for this size (first occurrence)
    regions = {}
    for rank, ts_df in timestamps.items():
        starts = ts_df.filter(
            (pl.col("where") == "START") & 
            (pl.col("size") == size)
        )
        stops = ts_df.filter(
            (pl.col("where") == "STOP") & 
            (pl.col("size") == size)
        )
        
        if len(starts) > 0:
            # Take the first (out-of-place) region
            regions[rank] = {
                "start": starts.row(0, named=True)["ts"],
                "stop": stops.row(0, named=True)["ts"],
            }
    
    if not regions:
        print(f"No regions found for size {size}")
        return
    
    # Find global time range
    min_ts = min(r["start"] for r in regions.values())
    max_ts = max(r["stop"] for r in regions.values())
    duration = max_ts - min_ts
    
    # Create time bins
    num_bins = 100
    bin_width = duration / num_bins
    
    # Count kernel activity in each bin for each rank
    ranks = sorted(regions.keys())
    activity_matrix = np.zeros((len(ranks), num_bins))
    
    for i, rank in enumerate(ranks):
        # Find PID for this rank
        pid = [p for p, r in pid_to_rank.items() if r == rank][0]
        
        if pid not in parquet_data:
            continue
        
        gpu_activity = parquet_data[pid].get("gpu_activity", pl.DataFrame())
        if len(gpu_activity) == 0:
            continue
        
        # Filter to region
        region = regions[rank]
        in_region = gpu_activity.filter(
            (pl.col("timestamp_ns") >= region["start"]) &
            (pl.col("timestamp_ns") <= region["stop"])
        )
        
        # Count activity in each bin
        for row in in_region.iter_rows(named=True):
            rel_start = row["timestamp_ns"] - min_ts
            rel_end = rel_start + row["duration_ns"]
            
            start_bin = max(0, int(rel_start / bin_width))
            end_bin = min(num_bins - 1, int(rel_end / bin_width))
            
            for b in range(start_bin, end_bin + 1):
                activity_matrix[i, b] += 1
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 4))
    
    im = ax.imshow(activity_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    
    ax.set_yticks(range(len(ranks)))
    ax.set_yticklabels([f"Rank {r}" for r in ranks])
    
    # X-axis labels (time in μs)
    time_labels = np.linspace(0, duration / 1000, 5)  # Convert to μs
    ax.set_xticks(np.linspace(0, num_bins - 1, 5))
    ax.set_xticklabels([f"{t:.0f}" for t in time_labels])
    ax.set_xlabel("Time (μs)")
    
    ax.set_title(f"Kernel Activity Across Ranks - Size: {size:,} bytes")
    
    cbar = fig.colorbar(im, ax=ax, label="Kernel Count")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"kernel_heatmap_{size}.png"), dpi=150)
    print(f"Saved: kernel_heatmap_{size}.png")
    plt.close()


def print_kernel_summary(kernel_stats: pl.DataFrame, categories: Dict[str, List[str]]):
    """Print a summary of kernel usage."""
    
    print("\n" + "="*80)
    print("KERNEL ANALYSIS SUMMARY")
    print("="*80 + "\n")
    
    # Total time breakdown by category
    print("TIME BREAKDOWN BY CATEGORY:")
    print("-" * 50)
    
    total_time = kernel_stats["total_duration_ns"].sum()
    
    for category, kernels in categories.items():
        if not kernels:
            continue
        
        cat_stats = kernel_stats.filter(pl.col("truncated_name").is_in(kernels))
        if len(cat_stats) > 0:
            cat_time = cat_stats["total_duration_ns"].sum()
            pct = (cat_time / total_time * 100) if total_time > 0 else 0
            print(f"  {category:25s}: {cat_time/1e6:10.2f} ms ({pct:5.1f}%)")
    
    print()
    
    # Top 10 kernels by time
    print("TOP 10 KERNELS BY TOTAL TIME:")
    print("-" * 50)
    
    top10 = kernel_stats.group_by("truncated_name").agg([
        pl.col("total_duration_ns").sum().alias("total_ns"),
        pl.col("count").sum().alias("total_count"),
    ]).sort("total_ns", descending=True).head(10)
    
    print(f"{'Kernel Name':<50} {'Time (ms)':>12} {'Count':>10}")
    print("-" * 74)
    for row in top10.iter_rows(named=True):
        name = row["truncated_name"][:47] + "..." if len(row["truncated_name"]) > 47 else row["truncated_name"]
        print(f"{name:<50} {row['total_ns']/1e6:>12.2f} {row['total_count']:>10}")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Advanced RCCL kernel analysis"
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
    parser.add_argument(
        "--heatmap-sizes",
        type=int,
        nargs="*",
        default=[1048576, 67108864],
        help="Sizes for kernel heatmaps"
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading data...")
    timestamps, parquet_data, pid_to_rank = load_all_data(
        args.timestamps, args.parquet_dir
    )
    
    print("\nAnalyzing kernel patterns...")
    kernel_stats = analyze_kernel_patterns(parquet_data, pid_to_rank)
    
    if len(kernel_stats) == 0:
        print("No kernel statistics collected!")
        return
    
    # Save kernel stats
    kernel_stats.write_parquet(os.path.join(args.output_dir, "kernel_statistics.parquet"))
    kernel_stats.write_csv(os.path.join(args.output_dir, "kernel_statistics.csv"))
    
    # Identify RCCL kernels
    categories = identify_rccl_kernels(kernel_stats)
    
    # Print summary
    print_kernel_summary(kernel_stats, categories)
    
    # Create plots
    print("Generating plots...")
    create_kernel_breakdown_plot(kernel_stats, args.output_dir)
    create_cross_rank_comparison(kernel_stats, args.output_dir)
    
    for size in args.heatmap_sizes:
        create_kernel_timeline_heatmap(
            timestamps, parquet_data, pid_to_rank,
            size, args.output_dir
        )
    
    print(f"\nAnalysis complete! Results in: {args.output_dir}")


if __name__ == "__main__":
    main()

