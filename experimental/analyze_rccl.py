#!/usr/bin/env python3
"""
RCCL Benchmark Analysis Tool

This script correlates RCCL benchmark timestamps with rocprofv3 Parquet output
to provide detailed analysis of GPU activity during collective operations.

The benchmark produces:
- rank_*_timestamps.csv: START/STOP timestamps for each benchmark region
- rocp/<host>/<pid>_parquet/: Parquet files with GPU activity data

Both use CLOCK_BOOTTIME timestamps in nanoseconds.
"""

import argparse
import glob
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import polars as pl
import matplotlib.pyplot as plt
import numpy as np

# Try to use seaborn for better styling
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="husl")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


@dataclass
class BenchmarkRegion:
    """A benchmark region (START to STOP) with metadata."""
    rank: int
    start_ts: int
    stop_ts: int
    test: str
    size: int
    place: str  # 'oop' or 'inp'
    dtype: str
    op: str
    root: int
    
    @property
    def duration_ns(self) -> int:
        return self.stop_ts - self.start_ts
    
    @property
    def duration_us(self) -> float:
        return self.duration_ns / 1000.0
    
    @property
    def duration_ms(self) -> float:
        return self.duration_ns / 1_000_000.0


def load_timestamp_csvs(pattern: str = "rank_*_timestamps.csv") -> Dict[int, pl.DataFrame]:
    """Load all rank timestamp CSVs into a dict keyed by rank."""
    result = {}
    for path in sorted(glob.glob(pattern)):
        df = pl.read_csv(path)
        # Extract rank from filename
        rank = int(Path(path).stem.split('_')[1])
        result[rank] = df
    return result


def load_parquet_data(parquet_dir: str) -> Dict[int, Dict[str, pl.DataFrame]]:
    """
    Load all Parquet data from the output directory.
    Returns dict keyed by PID, with sub-dict of DataFrame names.
    """
    result = {}
    
    # Find all *_parquet directories
    for pid_dir in sorted(glob.glob(os.path.join(parquet_dir, "*_parquet"))):
        pid = int(Path(pid_dir).name.replace("_parquet", ""))
        result[pid] = {}
        
        # Load timeline data
        gpu_activity_path = os.path.join(pid_dir, "timeline", "gpu_activity.parquet")
        if os.path.exists(gpu_activity_path):
            result[pid]["gpu_activity"] = pl.read_parquet(gpu_activity_path)
        
        # Load details
        kernel_info_path = os.path.join(pid_dir, "details", "kernel_info.parquet")
        if os.path.exists(kernel_info_path):
            result[pid]["kernel_info"] = pl.read_parquet(kernel_info_path)
        
        dispatch_info_path = os.path.join(pid_dir, "details", "dispatch_info.parquet")
        if os.path.exists(dispatch_info_path):
            result[pid]["dispatch_info"] = pl.read_parquet(dispatch_info_path)
        
        # Load reference data
        agents_path = os.path.join(pid_dir, "reference", "agents.parquet")
        if os.path.exists(agents_path):
            result[pid]["agents"] = pl.read_parquet(agents_path)
    
    return result


def parse_benchmark_regions(timestamps: Dict[int, pl.DataFrame]) -> Dict[int, List[BenchmarkRegion]]:
    """Convert timestamp CSVs into BenchmarkRegion objects per rank."""
    result = {}
    
    for rank, df in timestamps.items():
        regions = []
        # Pair up START and STOP events
        starts = df.filter(pl.col("where") == "START")
        stops = df.filter(pl.col("where") == "STOP")
        
        for i in range(len(starts)):
            start_row = starts.row(i, named=True)
            stop_row = stops.row(i, named=True)
            
            region = BenchmarkRegion(
                rank=rank,
                start_ts=start_row["ts"],
                stop_ts=stop_row["ts"],
                test=start_row["test"],
                size=start_row["size"],
                place=start_row["place"],
                dtype=start_row["type"],
                op=start_row["op"],
                root=start_row["root"],
            )
            regions.append(region)
        
        result[rank] = regions
    
    return result


def correlate_gpu_activity(
    region: BenchmarkRegion,
    gpu_activity: pl.DataFrame,
    kernel_info: pl.DataFrame,
) -> pl.DataFrame:
    """
    Find all GPU activity that falls within a benchmark region.
    Returns enriched DataFrame with kernel names.
    """
    # Filter GPU activity by timestamp range
    in_region = gpu_activity.filter(
        (pl.col("timestamp_ns") >= region.start_ts) &
        (pl.col("timestamp_ns") + pl.col("duration_ns") <= region.stop_ts)
    )
    
    # Join with kernel info to get names
    if len(in_region) > 0 and len(kernel_info) > 0:
        in_region = in_region.join(
            kernel_info.select(["kernel_id", "kernel_name", "truncated_name"]),
            on="kernel_id",
            how="left"
        )
    
    return in_region


def compute_region_statistics(
    regions: Dict[int, List[BenchmarkRegion]],
    parquet_data: Dict[int, Dict[str, pl.DataFrame]],
    pid_to_rank: Dict[int, int],
) -> pl.DataFrame:
    """
    Compute statistics for each benchmark region across all ranks.
    """
    stats = []
    
    for rank, rank_regions in regions.items():
        # Find the PID for this rank
        pid = None
        for p, r in pid_to_rank.items():
            if r == rank:
                pid = p
                break
        
        if pid is None or pid not in parquet_data:
            continue
        
        pdata = parquet_data[pid]
        gpu_activity = pdata.get("gpu_activity", pl.DataFrame())
        kernel_info = pdata.get("kernel_info", pl.DataFrame())
        
        for region in rank_regions:
            # Get GPU activity in this region
            activity = correlate_gpu_activity(region, gpu_activity, kernel_info)
            
            # Compute statistics
            total_gpu_ns = activity["duration_ns"].sum() if len(activity) > 0 else 0
            num_kernels = len(activity)
            
            # Kernel breakdown
            kernel_counts = {}
            if len(activity) > 0 and "truncated_name" in activity.columns:
                for name in activity["truncated_name"].unique().to_list():
                    if name:
                        count = activity.filter(pl.col("truncated_name") == name).height
                        kernel_counts[name] = count
            
            stats.append({
                "rank": rank,
                "test": region.test,
                "size": region.size,
                "place": region.place,
                "dtype": region.dtype,
                "op": region.op,
                "wall_time_us": region.duration_us,
                "gpu_time_us": total_gpu_ns / 1000.0,
                "gpu_utilization": (total_gpu_ns / region.duration_ns * 100) if region.duration_ns > 0 else 0,
                "num_kernels": num_kernels,
                "avg_kernel_us": (total_gpu_ns / num_kernels / 1000.0) if num_kernels > 0 else 0,
            })
    
    return pl.DataFrame(stats)


def create_size_scaling_plot(
    stats: pl.DataFrame,
    output_dir: str,
    test_name: str = "AllReduce",
):
    """Create a plot showing performance scaling with message size."""
    
    # Filter for the specific test and aggregate across ranks
    filtered = stats.filter(pl.col("test") == test_name)
    
    if len(filtered) == 0:
        print(f"No data for test: {test_name}")
        return
    
    # Aggregate by size and place
    agg = filtered.group_by(["size", "place"]).agg([
        pl.col("wall_time_us").mean().alias("avg_wall_time_us"),
        pl.col("wall_time_us").std().alias("std_wall_time_us"),
        pl.col("gpu_time_us").mean().alias("avg_gpu_time_us"),
        pl.col("num_kernels").mean().alias("avg_num_kernels"),
        pl.col("gpu_utilization").mean().alias("avg_gpu_util"),
    ]).sort("size")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Wall time vs size
    ax = axes[0, 0]
    for place in ["oop", "inp"]:
        data = agg.filter(pl.col("place") == place)
        if len(data) > 0:
            sizes = data["size"].to_numpy()
            times = data["avg_wall_time_us"].to_numpy()
            label = "Out-of-place" if place == "oop" else "In-place"
            ax.loglog(sizes, times, 'o-', label=label, markersize=6)
    
    ax.set_xlabel("Message Size (bytes)")
    ax.set_ylabel("Wall Time (μs)")
    ax.set_title(f"{test_name} - Wall Time Scaling")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: GPU utilization vs size
    ax = axes[0, 1]
    for place in ["oop", "inp"]:
        data = agg.filter(pl.col("place") == place)
        if len(data) > 0:
            sizes = data["size"].to_numpy()
            util = data["avg_gpu_util"].to_numpy()
            label = "Out-of-place" if place == "oop" else "In-place"
            ax.semilogx(sizes, util, 'o-', label=label, markersize=6)
    
    ax.set_xlabel("Message Size (bytes)")
    ax.set_ylabel("GPU Utilization (%)")
    ax.set_title(f"{test_name} - GPU Utilization")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Plot 3: Bandwidth vs size
    ax = axes[1, 0]
    for place in ["oop", "inp"]:
        data = agg.filter(pl.col("place") == place)
        if len(data) > 0:
            sizes = data["size"].to_numpy().astype(float)
            times = data["avg_wall_time_us"].to_numpy()
            # Bandwidth in GB/s: (size in bytes) / (time in us) = MB/s, divide by 1000 for GB/s
            bw = sizes / times / 1000.0
            label = "Out-of-place" if place == "oop" else "In-place"
            ax.semilogx(sizes, bw, 'o-', label=label, markersize=6)
    
    ax.set_xlabel("Message Size (bytes)")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title(f"{test_name} - Algorithm Bandwidth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Number of kernels vs size
    ax = axes[1, 1]
    for place in ["oop", "inp"]:
        data = agg.filter(pl.col("place") == place)
        if len(data) > 0:
            sizes = data["size"].to_numpy()
            num_k = data["avg_num_kernels"].to_numpy()
            label = "Out-of-place" if place == "oop" else "In-place"
            ax.semilogx(sizes, num_k, 'o-', label=label, markersize=6)
    
    ax.set_xlabel("Message Size (bytes)")
    ax.set_ylabel("Number of Kernel Dispatches")
    ax.set_title(f"{test_name} - Kernel Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{test_name.lower()}_scaling.png"), dpi=150)
    plt.savefig(os.path.join(output_dir, f"{test_name.lower()}_scaling.pdf"))
    print(f"Saved: {test_name.lower()}_scaling.png/pdf")
    plt.close()


def create_timeline_plot(
    region: BenchmarkRegion,
    activity: pl.DataFrame,
    output_path: str,
):
    """Create a timeline visualization for a single benchmark region."""
    if len(activity) == 0:
        print(f"No GPU activity for region: size={region.size}, place={region.place}")
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Normalize timestamps to start of region
    start_ts = region.start_ts
    
    # Color by kernel type
    colors = plt.cm.tab20.colors
    kernel_names = activity["truncated_name"].unique().to_list() if "truncated_name" in activity.columns else []
    name_to_color = {name: colors[i % len(colors)] for i, name in enumerate(kernel_names)}
    
    for i, row in enumerate(activity.iter_rows(named=True)):
        x_start = (row["timestamp_ns"] - start_ts) / 1000.0  # Convert to μs
        width = row["duration_ns"] / 1000.0
        name = row.get("truncated_name", "Unknown")
        color = name_to_color.get(name, "gray")
        
        ax.barh(0, width, left=x_start, height=0.5, color=color, edgecolor='black', linewidth=0.5)
    
    # Mark region boundaries
    ax.axvline(0, color='green', linestyle='--', linewidth=2, label='Region START')
    ax.axvline(region.duration_us, color='red', linestyle='--', linewidth=2, label='Region STOP')
    
    ax.set_xlabel("Time (μs)")
    ax.set_title(f"{region.test} - Size: {region.size:,} bytes, Place: {region.place}")
    ax.set_yticks([])
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def create_rank_comparison_plot(
    stats: pl.DataFrame,
    output_dir: str,
    size: int,
):
    """Create a plot comparing performance across ranks for a specific size."""
    
    filtered = stats.filter(pl.col("size") == size)
    
    if len(filtered) == 0:
        print(f"No data for size: {size}")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    ranks = sorted(filtered["rank"].unique().to_list())
    
    # Wall time comparison
    ax = axes[0]
    for place in ["oop", "inp"]:
        data = filtered.filter(pl.col("place") == place)
        if len(data) > 0:
            times = [data.filter(pl.col("rank") == r)["wall_time_us"].to_list()[0] 
                     for r in ranks if len(data.filter(pl.col("rank") == r)) > 0]
            label = "Out-of-place" if place == "oop" else "In-place"
            ax.bar([r + (0.2 if place == "oop" else -0.2) for r in range(len(ranks))], 
                   times, width=0.4, label=label)
    
    ax.set_xlabel("Rank")
    ax.set_ylabel("Wall Time (μs)")
    ax.set_title(f"Wall Time - Size: {size:,} bytes")
    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels(ranks)
    ax.legend()
    
    # GPU utilization comparison
    ax = axes[1]
    for place in ["oop", "inp"]:
        data = filtered.filter(pl.col("place") == place)
        if len(data) > 0:
            utils = [data.filter(pl.col("rank") == r)["gpu_utilization"].to_list()[0] 
                     for r in ranks if len(data.filter(pl.col("rank") == r)) > 0]
            label = "Out-of-place" if place == "oop" else "In-place"
            ax.bar([r + (0.2 if place == "oop" else -0.2) for r in range(len(ranks))], 
                   utils, width=0.4, label=label)
    
    ax.set_xlabel("Rank")
    ax.set_ylabel("GPU Utilization (%)")
    ax.set_title(f"GPU Utilization - Size: {size:,} bytes")
    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels(ranks)
    ax.legend()
    ax.set_ylim(0, 100)
    
    # Kernel count comparison
    ax = axes[2]
    for place in ["oop", "inp"]:
        data = filtered.filter(pl.col("place") == place)
        if len(data) > 0:
            counts = [data.filter(pl.col("rank") == r)["num_kernels"].to_list()[0] 
                      for r in ranks if len(data.filter(pl.col("rank") == r)) > 0]
            label = "Out-of-place" if place == "oop" else "In-place"
            ax.bar([r + (0.2 if place == "oop" else -0.2) for r in range(len(ranks))], 
                   counts, width=0.4, label=label)
    
    ax.set_xlabel("Rank")
    ax.set_ylabel("Number of Kernels")
    ax.set_title(f"Kernel Count - Size: {size:,} bytes")
    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels(ranks)
    ax.legend()
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"rank_comparison_{size}.png")
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")
    plt.close()


def print_summary_statistics(stats: pl.DataFrame):
    """Print summary statistics to console."""
    print("\n" + "="*80)
    print("RCCL BENCHMARK ANALYSIS SUMMARY")
    print("="*80 + "\n")
    
    # Overall statistics
    print("OVERALL STATISTICS:")
    print("-" * 40)
    print(f"  Total benchmark regions: {len(stats)}")
    print(f"  Number of ranks: {stats['rank'].n_unique()}")
    print(f"  Message sizes tested: {sorted(stats['size'].unique().to_list())}")
    print(f"  Collective operations: {stats['test'].unique().to_list()}")
    print()
    
    # Per-size statistics
    print("PER-SIZE STATISTICS (averaged across ranks):")
    print("-" * 40)
    
    size_stats = stats.group_by("size").agg([
        pl.col("wall_time_us").mean().alias("avg_time_us"),
        pl.col("wall_time_us").std().alias("std_time_us"),
        pl.col("gpu_utilization").mean().alias("avg_gpu_util"),
        pl.col("num_kernels").mean().alias("avg_kernels"),
    ]).sort("size")
    
    print(f"{'Size':>12} {'Avg Time (μs)':>14} {'Std (μs)':>12} {'GPU Util %':>12} {'Avg Kernels':>12}")
    print("-" * 64)
    for row in size_stats.iter_rows(named=True):
        std = row["std_time_us"] if row["std_time_us"] is not None else 0
        print(f"{row['size']:>12,} {row['avg_time_us']:>14.2f} {std:>12.2f} {row['avg_gpu_util']:>12.1f} {row['avg_kernels']:>12.1f}")
    
    print()
    
    # Bandwidth analysis
    print("BANDWIDTH ANALYSIS:")
    print("-" * 40)
    
    # Add bandwidth column
    bw_stats = stats.with_columns([
        (pl.col("size") / pl.col("wall_time_us") / 1000.0).alias("bw_gbps")
    ])
    
    max_bw = bw_stats.select(pl.col("bw_gbps").max()).item()
    max_bw_row = bw_stats.filter(pl.col("bw_gbps") == max_bw).row(0, named=True)
    
    print(f"  Peak bandwidth: {max_bw:.2f} GB/s at size {max_bw_row['size']:,} bytes")
    
    # Large message bandwidth (>1MB)
    large_msg = bw_stats.filter(pl.col("size") >= 1024*1024)
    if len(large_msg) > 0:
        avg_large_bw = large_msg["bw_gbps"].mean()
        print(f"  Average bandwidth (>= 1MB): {avg_large_bw:.2f} GB/s")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze RCCL benchmark results with rocprofv3 Parquet data"
    )
    parser.add_argument(
        "--timestamps", "-t",
        default="rank_*_timestamps.csv",
        help="Glob pattern for timestamp CSV files (default: rank_*_timestamps.csv)"
    )
    parser.add_argument(
        "--parquet-dir", "-p",
        required=True,
        help="Directory containing Parquet output (e.g., rocp/ringo)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="analysis_output",
        help="Directory for output plots and data (default: analysis_output)"
    )
    parser.add_argument(
        "--timeline-sizes",
        type=int,
        nargs="*",
        help="Sizes (in bytes) to generate timeline plots for"
    )
    parser.add_argument(
        "--rank-comparison-sizes",
        type=int,
        nargs="*",
        help="Sizes (in bytes) to generate rank comparison plots for"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading timestamp CSVs...")
    timestamps = load_timestamp_csvs(args.timestamps)
    print(f"  Loaded data for {len(timestamps)} ranks")
    
    print("Loading Parquet data...")
    parquet_data = load_parquet_data(args.parquet_dir)
    print(f"  Loaded data for {len(parquet_data)} processes")
    
    # Create PID to rank mapping
    # The PIDs are ordered, and we assume they correspond to ranks in order
    pids = sorted(parquet_data.keys())
    pid_to_rank = {pid: i for i, pid in enumerate(pids)}
    print(f"  PID to rank mapping: {pid_to_rank}")
    
    print("\nParsing benchmark regions...")
    regions = parse_benchmark_regions(timestamps)
    total_regions = sum(len(r) for r in regions.values())
    print(f"  Found {total_regions} benchmark regions")
    
    print("\nComputing statistics...")
    stats = compute_region_statistics(regions, parquet_data, pid_to_rank)
    
    # Save statistics
    stats_file = os.path.join(args.output_dir, "benchmark_statistics.parquet")
    stats.write_parquet(stats_file)
    print(f"  Saved statistics to: {stats_file}")
    
    # Also save as CSV for easy viewing
    stats_csv = os.path.join(args.output_dir, "benchmark_statistics.csv")
    stats.write_csv(stats_csv)
    print(f"  Saved statistics to: {stats_csv}")
    
    # Print summary
    print_summary_statistics(stats)
    
    # Create plots
    print("\nGenerating plots...")
    
    # Size scaling plot
    test_names = stats["test"].unique().to_list()
    for test_name in test_names:
        create_size_scaling_plot(stats, args.output_dir, test_name)
    
    # Rank comparison plots
    if args.rank_comparison_sizes:
        for size in args.rank_comparison_sizes:
            create_rank_comparison_plot(stats, args.output_dir, size)
    else:
        # Default: create for a few representative sizes
        sizes = sorted(stats["size"].unique().to_list())
        if len(sizes) >= 3:
            selected = [sizes[0], sizes[len(sizes)//2], sizes[-1]]
            for size in selected:
                create_rank_comparison_plot(stats, args.output_dir, size)
    
    # Timeline plots
    if args.timeline_sizes:
        print("\nGenerating timeline plots...")
        for size in args.timeline_sizes:
            for rank in sorted(regions.keys()):
                for region in regions[rank]:
                    if region.size == size:
                        # Find PID for this rank
                        pid = [p for p, r in pid_to_rank.items() if r == rank][0]
                        if pid in parquet_data:
                            pdata = parquet_data[pid]
                            activity = correlate_gpu_activity(
                                region,
                                pdata.get("gpu_activity", pl.DataFrame()),
                                pdata.get("kernel_info", pl.DataFrame()),
                            )
                            output_path = os.path.join(
                                args.output_dir,
                                f"timeline_rank{rank}_size{size}_{region.place}.png"
                            )
                            create_timeline_plot(region, activity, output_path)
                            print(f"  Saved: {output_path}")
    
    print("\nAnalysis complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

