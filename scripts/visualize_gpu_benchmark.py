#!/usr/bin/env python3
"""
Visualize raw GPU interconnect benchmark data
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import sys
from pathlib import Path

def parse_gpu_benchmark_data(content):
    """Parse GPU transfer benchmark output into structured data"""
    results = []

    lines = content.split('\n')
    is_latency = False
    is_bandwidth = False

    for line in lines:
        line = line.strip()

        # Skip empty lines and headers
        if not line or line.startswith('=') or line.startswith('-'):
            continue

        # Detect section headers
        if 'LATENCY BENCHMARKS' in line:
            is_latency = True
            is_bandwidth = False
            continue
        elif 'BANDWIDTH BENCHMARKS' in line:
            is_latency = False
            is_bandwidth = True
            continue

        # Skip table headers
        if 'Size' in line and 'GPU Pair' in line:
            continue

        # Parse data lines - look for lines with → and numbers
        if '→' in line and any(char.isdigit() for char in line):
            try:
                parts = line.split()

                # Find the GPU pair (contains →)
                gpu_pair_idx = None
                for i, part in enumerate(parts):
                    if '→' in part:
                        gpu_pair_idx = i
                        break

                if gpu_pair_idx is not None:
                    gpu_pair = parts[gpu_pair_idx]

                    # Parse size (first column)
                    size_str = parts[0]
                    if size_str.endswith('KB'):
                        size_bytes = int(size_str[:-2]) * 1024
                    elif size_str.endswith('MB'):
                        size_bytes = int(size_str[:-2]) * 1024 * 1024
                    elif size_str.endswith('GB'):
                        size_bytes = int(size_str[:-2]) * 1024 * 1024 * 1024
                    elif size_str.endswith('B'):
                        size_bytes = int(size_str[:-1])
                    else:
                        size_bytes = int(size_str)

                    # Parse metrics based on section
                    if is_latency:
                        # Format: Size GPU_Pair Latency Bandwidth
                        if len(parts) >= 4:
                            latency_us = float(parts[2])
                            bandwidth_gbps = float(parts[3])
                        else:
                            continue
                    elif is_bandwidth:
                        # Format: Size GPU_Pair Avg_Latency Bandwidth
                        if len(parts) >= 4 and parts[2].replace('.', '').isdigit():
                            latency_us = float(parts[2])  # Avg latency
                            bandwidth_gbps = float(parts[3])
                        else:
                            continue
                    else:
                        continue

                    results.append({
                        'gpu_pair': gpu_pair,
                        'size_bytes': size_bytes,
                        'latency_us': latency_us,
                        'bandwidth_gbps': bandwidth_gbps,
                        'test_type': 'latency' if is_latency else 'bandwidth'
                    })

            except (ValueError, IndexError) as e:
                continue

    return results

def format_bytes(bytes_val):
    """Format bytes in human-readable form"""
    if bytes_val >= 1024 * 1024 * 1024:
        return ".1f"
    elif bytes_val >= 1024 * 1024:
        return ".1f"
    elif bytes_val >= 1024:
        return ".1f"
    else:
        return f"{bytes_val}B"

def create_gpu_benchmark_visualizations(data):
    """Create comprehensive visualizations of GPU benchmark data"""

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Split into latency and bandwidth data
    latency_df = df[df['test_type'] == 'latency']
    bandwidth_df = df[df['test_type'] == 'bandwidth']

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))

    # Define color scheme for GPU pairs
    gpu_pairs = sorted(df['gpu_pair'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(gpu_pairs)))
    pair_colors = dict(zip(gpu_pairs, colors))

    # Plot 1: Latency vs Message Size for all GPU pairs
    ax1 = plt.subplot(2, 3, 1)
    for pair in gpu_pairs:
        pair_data = latency_df[latency_df['gpu_pair'] == pair]
        if len(pair_data) > 0:
            ax1.plot(pair_data['size_bytes'], pair_data['latency_us'],
                    'o-', label=pair, color=pair_colors[pair], markersize=4, linewidth=1.5)

    ax1.set_xscale('log', base=2)
    ax1.set_xlabel('Message Size (Bytes)')
    ax1.set_ylabel('Latency (μs)')
    ax1.set_title('GPU Interconnect Latency\n(All Pairs)')
    ax1.grid(True, alpha=0.3)

    # Format X axis
    x_ticks = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([format_bytes(x) for x in x_ticks], rotation=45)

    # Plot 2: Latency distribution (box plot)
    ax2 = plt.subplot(2, 3, 2)
    latency_pivot = latency_df.pivot(index='size_bytes', columns='gpu_pair', values='latency_us')
    latency_pivot.boxplot(ax=ax2, patch_artist=True)
    ax2.set_xlabel('Message Size')
    ax2.set_ylabel('Latency (μs)')
    ax2.set_title('Latency Distribution\nAcross GPU Pairs')
    # Set ticks and labels to match
    ax2.set_xticks(range(1, len(latency_pivot.index) + 1))
    ax2.set_xticklabels([format_bytes(x) for x in latency_pivot.index], rotation=45)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Average latency by GPU pair
    ax3 = plt.subplot(2, 3, 3)
    avg_latency = latency_df.groupby('gpu_pair')['latency_us'].mean().sort_values()
    colors_avg = [pair_colors[pair] for pair in avg_latency.index]
    avg_latency.plot(kind='bar', ax=ax3, color=colors_avg)
    ax3.set_xlabel('GPU Pair')
    ax3.set_ylabel('Average Latency (μs)')
    ax3.set_title('Average Latency by GPU Pair')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Bandwidth vs Message Size for all GPU pairs
    ax4 = plt.subplot(2, 3, 4)
    for pair in gpu_pairs:
        pair_data = bandwidth_df[bandwidth_df['gpu_pair'] == pair]
        if len(pair_data) > 0:
            ax4.plot(pair_data['size_bytes'], pair_data['bandwidth_gbps'],
                    's-', label=pair, color=pair_colors[pair], markersize=4, linewidth=1.5)

    ax4.set_xscale('log', base=2)
    ax4.set_xlabel('Message Size (Bytes)')
    ax4.set_ylabel('Bandwidth (GB/s)')
    ax4.set_title('GPU Interconnect Bandwidth\n(All Pairs)')
    ax4.grid(True, alpha=0.3)

    # Format X axis for bandwidth
    x_ticks_bw = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144,
                  524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432,
                  67108864, 134217728, 268435456, 536870912, 1073741824]
    ax4.set_xticks(x_ticks_bw)
    ax4.set_xticklabels([format_bytes(x) for x in x_ticks_bw], rotation=45)

    # Plot 5: Bandwidth distribution (box plot)
    ax5 = plt.subplot(2, 3, 5)
    # Only show for larger message sizes where bandwidth is meaningful
    large_bw_df = bandwidth_df[bandwidth_df['size_bytes'] >= 65536]  # 64KB+
    if len(large_bw_df) > 0:
        bw_pivot = large_bw_df.pivot(index='size_bytes', columns='gpu_pair', values='bandwidth_gbps')
        bw_pivot.boxplot(ax=ax5, patch_artist=True)
        ax5.set_xlabel('Message Size')
        ax5.set_ylabel('Bandwidth (GB/s)')
        ax5.set_title('Bandwidth Distribution\n(Large Messages)')
        # Set ticks and labels to match
        ax5.set_xticks(range(1, len(bw_pivot.index) + 1))
        ax5.set_xticklabels([format_bytes(x) for x in bw_pivot.index], rotation=45)
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'No large message data', ha='center', va='center', transform=ax5.transAxes)

    # Plot 6: Average bandwidth by GPU pair (large messages)
    ax6 = plt.subplot(2, 3, 6)
    if len(large_bw_df) > 0:
        avg_bandwidth = large_bw_df.groupby('gpu_pair')['bandwidth_gbps'].mean().sort_values(ascending=False)
        colors_bw = [pair_colors[pair] for pair in avg_bandwidth.index]
        avg_bandwidth.plot(kind='bar', ax=ax6, color=colors_bw)
        ax6.set_xlabel('GPU Pair')
        ax6.set_ylabel('Average Bandwidth (GB/s)')
        ax6.set_title('Average Bandwidth by GPU Pair\n(Large Messages)')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3, axis='y')

    # Add legend to first plot only
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    plt.tight_layout()
    return fig

def create_summary_statistics(data):
    """Create summary statistics plots"""

    df = pd.DataFrame(data)
    latency_df = df[df['test_type'] == 'latency']
    bandwidth_df = df[df['test_type'] == 'bandwidth']

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Latency statistics across message sizes
    latency_stats = latency_df.groupby('size_bytes')['latency_us'].agg(['mean', 'std', 'min', 'max'])
    ax1.errorbar(latency_stats.index, latency_stats['mean'], yerr=latency_stats['std'],
                fmt='o-', capsize=5, label='Mean ± Std')
    ax1.plot(latency_stats.index, latency_stats['min'], 'g--', alpha=0.7, label='Min/Max')
    ax1.plot(latency_stats.index, latency_stats['max'], 'g--', alpha=0.7)
    ax1.fill_between(latency_stats.index, latency_stats['min'], latency_stats['max'],
                    alpha=0.2, color='green', label='Range')

    ax1.set_xscale('log', base=2)
    ax1.set_xlabel('Message Size (Bytes)')
    ax1.set_ylabel('Latency (μs)')
    ax1.set_title('Latency Statistics Across Message Sizes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    x_ticks = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([format_bytes(x) for x in x_ticks], rotation=45)

    # Plot 2: Bandwidth statistics across message sizes
    bandwidth_stats = bandwidth_df.groupby('size_bytes')['bandwidth_gbps'].agg(['mean', 'std', 'min', 'max'])
    ax2.errorbar(bandwidth_stats.index, bandwidth_stats['mean'], yerr=bandwidth_stats['std'],
                fmt='s-', capsize=5, label='Mean ± Std')
    ax2.plot(bandwidth_stats.index, bandwidth_stats['min'], 'r--', alpha=0.7, label='Min/Max')
    ax2.plot(bandwidth_stats.index, bandwidth_stats['max'], 'r--', alpha=0.7)
    ax2.fill_between(bandwidth_stats.index, bandwidth_stats['min'], bandwidth_stats['max'],
                    alpha=0.2, color='red', label='Range')

    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('Message Size (Bytes)')
    ax2.set_ylabel('Bandwidth (GB/s)')
    ax2.set_title('Bandwidth Statistics Across Message Sizes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: GPU pair performance comparison (latency)
    latency_by_pair = latency_df.groupby('gpu_pair')['latency_us'].describe()
    latency_by_pair = latency_by_pair.sort_values('mean')

    ax3.bar(range(len(latency_by_pair)), latency_by_pair['mean'],
           yerr=latency_by_pair['std'], capsize=5, alpha=0.7)
    ax3.set_xlabel('GPU Pair (sorted by mean latency)')
    ax3.set_ylabel('Latency (μs)')
    ax3.set_title('GPU Pair Latency Performance')
    ax3.set_xticks(range(len(latency_by_pair)))
    ax3.set_xticklabels(latency_by_pair.index, rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: GPU pair performance comparison (bandwidth)
    large_bw_df = bandwidth_df[bandwidth_df['size_bytes'] >= 65536]  # 64KB+
    if len(large_bw_df) > 0:
        bandwidth_by_pair = large_bw_df.groupby('gpu_pair')['bandwidth_gbps'].describe()
        bandwidth_by_pair = bandwidth_by_pair.sort_values('mean', ascending=False)

        ax4.bar(range(len(bandwidth_by_pair)), bandwidth_by_pair['mean'],
               yerr=bandwidth_by_pair['std'], capsize=5, alpha=0.7)
        ax4.set_xlabel('GPU Pair (sorted by mean bandwidth)')
        ax4.set_ylabel('Bandwidth (GB/s)')
        ax4.set_title('GPU Pair Bandwidth Performance\n(Large Messages)')
        ax4.set_xticks(range(len(bandwidth_by_pair)))
        ax4.set_xticklabels(bandwidth_by_pair.index, rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'No large message data', ha='center', va='center', transform=ax4.transAxes)

    plt.tight_layout()
    return fig

def print_data_summary(data):
    """Print summary statistics of the benchmark data"""

    df = pd.DataFrame(data)
    latency_df = df[df['test_type'] == 'latency']
    bandwidth_df = df[df['test_type'] == 'bandwidth']

    print("\n" + "="*80)
    print(" GPU INTERCONNECT BENCHMARK DATA SUMMARY")
    print("="*80)

    print("\nDATA OVERVIEW:")
    print(f"  Total data points: {len(df)}")
    print(f"  Latency measurements: {len(latency_df)}")
    print(f"  Bandwidth measurements: {len(bandwidth_df)}")
    print(f"  GPU pairs tested: {len(df['gpu_pair'].unique())}")

    print("\nLATENCY STATISTICS:")
    print(f"  Message sizes: {latency_df['size_bytes'].min()} - {latency_df['size_bytes'].max()} bytes")
    print(f"  Latency range: {latency_df['latency_us'].min():.3f} - {latency_df['latency_us'].max():.3f} μs")
    print(f"  Average latency: {latency_df['latency_us'].mean():.3f} μs")
    print(f"  Measurements per pair: {len(latency_df) // len(latency_df['gpu_pair'].unique())}")

    print("\nBANDWIDTH STATISTICS:")
    print(f"  Message sizes: {bandwidth_df['size_bytes'].min()} - {bandwidth_df['size_bytes'].max()} bytes")
    print(f"  Bandwidth range: {bandwidth_df['bandwidth_gbps'].min():.3f} - {bandwidth_df['bandwidth_gbps'].max():.3f} GB/s")
    print(f"  Average bandwidth: {bandwidth_df['bandwidth_gbps'].mean():.3f} GB/s")
    print(f"  Measurements per pair: {len(bandwidth_df) // len(bandwidth_df['gpu_pair'].unique())}")

    # Best and worst performing pairs
    if len(latency_df) > 0:
        latency_by_pair = latency_df.groupby('gpu_pair')['latency_us'].mean()
        best_latency_pair = latency_by_pair.idxmin()
        worst_latency_pair = latency_by_pair.idxmax()

        print("\nPERFORMANCE VARIATION:")
        print(f"  Best latency pair: {best_latency_pair} ({latency_by_pair[best_latency_pair]:.3f} μs)")
        print(f"  Worst latency pair: {worst_latency_pair} ({latency_by_pair[worst_latency_pair]:.3f} μs)")
        print(f"  Latency variation: {((latency_by_pair.max() - latency_by_pair.min()) / latency_by_pair.mean() * 100):.1f}%")

    if len(bandwidth_df[bandwidth_df['size_bytes'] >= 65536]) > 0:
        large_bw = bandwidth_df[bandwidth_df['size_bytes'] >= 65536]
        bandwidth_by_pair = large_bw.groupby('gpu_pair')['bandwidth_gbps'].mean()
        best_bw_pair = bandwidth_by_pair.idxmax()
        worst_bw_pair = bandwidth_by_pair.idxmin()

        print(f"  Best bandwidth pair: {best_bw_pair} ({bandwidth_by_pair[best_bw_pair]:.3f} GB/s)")
        print(f"  Worst bandwidth pair: {worst_bw_pair} ({bandwidth_by_pair[worst_bw_pair]:.3f} GB/s)")
        print(f"  Bandwidth variation: {((bandwidth_by_pair.max() - bandwidth_by_pair.min()) / bandwidth_by_pair.mean() * 100):.1f}%")

    print("\n" + "="*80)

def main():
    # File paths
    gpu_file = Path("gpu_transfer_results.txt")

    if not gpu_file.exists():
        print(f"Error: GPU benchmark results not found: {gpu_file}")
        return 1

    # Load GPU benchmark data
    print(f"Loading GPU benchmark results from {gpu_file}")
    with open(gpu_file, 'r') as f:
        gpu_content = f.read()

    gpu_data = parse_gpu_benchmark_data(gpu_content)
    print(f"Parsed {len(gpu_data)} GPU benchmark data points")

    if len(gpu_data) == 0:
        print("Error: No data parsed from GPU benchmark file")
        return 1

    # Print summary statistics
    print_data_summary(gpu_data)

    # Create comprehensive visualizations
    print("\nCreating comprehensive GPU benchmark visualizations...")

    # Main visualization
    fig1 = create_gpu_benchmark_visualizations(gpu_data)
    output_file1 = "gpu_benchmark_detailed.png"
    fig1.savefig(output_file1, dpi=150, bbox_inches='tight')
    print(f"Detailed plots saved to: {output_file1}")

    # Summary statistics
    fig2 = create_summary_statistics(gpu_data)
    output_file2 = "gpu_benchmark_summary.png"
    fig2.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"Summary plots saved to: {output_file2}")

    # Copy to web server
    import shutil
    web_path = Path("build/profile_results_all_ranks/all_reduce_perf_2ranks_20251031_153425")
    if web_path.exists():
        shutil.copy(output_file1, web_path)
        shutil.copy(output_file2, web_path)
        print(f"Plots copied to web server:")
        print(f"  Detailed: http://localhost:8080/{output_file1}")
        print(f"  Summary:  http://localhost:8080/{output_file2}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
