#!/usr/bin/env python3
"""
Create boxplots for individual kernel timings from RCCL benchmark data.

This script creates boxplot visualizations showing the distribution of
individual kernel execution times across all iterations and ranks.

Usage:
    python create_boxplots.py <run_directory>

Example:
    python create_boxplots.py /work/lmeadows/rccl/data/hostname/run_all_reduce_20251104_165031
"""

import argparse
import os
import sys
import glob
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_timing_data(run_dir):
    """Load all_rank*.csv files from run directory."""
    timing_files = glob.glob(os.path.join(run_dir, 'all_rank*.csv'))
    
    if not timing_files:
        return None
    
    dfs = []
    for filepath in timing_files:
        try:
            df = pd.read_csv(filepath)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
    
    if not dfs:
        return None
    
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def load_bic_segmentation(run_dir, benchmark_name):
    """Load BIC segmentation JSON if available."""
    seg_file = os.path.join(run_dir, f'{benchmark_name}_bic_segmentation.json')
    
    if not os.path.exists(seg_file):
        return None
    
    try:
        with open(seg_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load segmentation: {e}")
        return None


def format_size(size_bytes):
    """Format size in human-readable form."""
    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.1f} GiB"
    elif size_bytes >= 1024**2:
        return f"{size_bytes / (1024**2):.0f} MiB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.0f} KiB"
    else:
        return f"{size_bytes} B"


def create_segment_boxplots(timing_df, segmentation, benchmark_name, output_dir):
    """Create boxplots per BIC segment."""
    
    if segmentation is None:
        print("Warning: No segmentation data available, creating single plot")
        segments = [{
            'segment': 0,
            'size_range_bytes': [timing_df['size_bytes'].min(), timing_df['size_bytes'].max()],
            'model': 'all'
        }]
    else:
        segments = segmentation['segments']
    
    output_files = []
    
    for seg in segments:
        seg_num = seg['segment']
        size_min, size_max = seg['size_range_bytes']
        
        # Filter data for this segment
        seg_data = timing_df[
            (timing_df['size_bytes'] >= size_min) & 
            (timing_df['size_bytes'] <= size_max)
        ].copy()
        
        if len(seg_data) == 0:
            continue
        
        # Convert to microseconds
        seg_data['time_us'] = seg_data['time_seconds'] * 1e6
        
        # Get unique sizes in this segment
        sizes = sorted(seg_data['size_bytes'].unique())
        n_sizes = len(sizes)
        
        if n_sizes == 0:
            continue
        
        # Create figure - use 2 rows if more than 7 sizes
        if n_sizes > 7:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(14, n_sizes), 12))
            axes = [ax1, ax2]
            # Split sizes between two rows
            mid = (n_sizes + 1) // 2
            sizes_per_ax = [sizes[:mid], sizes[mid:]]
        else:
            fig, ax = plt.subplots(1, 1, figsize=(max(12, n_sizes * 1.5), 6))
            axes = [ax]
            sizes_per_ax = [sizes]
        
        # Plot each row
        for ax_idx, (ax, size_list) in enumerate(zip(axes, sizes_per_ax)):
            plot_data = []
            labels = []
            colors = []
            positions = []
            pos = 1
            
            for size in size_list:
                size_data = seg_data[seg_data['size_bytes'] == size]
                
                # Out-of-place
                oop_data = size_data[size_data['inplace'] == 0]['time_us'].values
                if len(oop_data) > 0:
                    plot_data.append(oop_data)
                    labels.append(f"{format_size(size)}\nOOP")
                    colors.append('lightblue')
                    positions.append(pos)
                    pos += 1
                
                # In-place
                inp_data = size_data[size_data['inplace'] == 1]['time_us'].values
                if len(inp_data) > 0:
                    plot_data.append(inp_data)
                    labels.append(f"{format_size(size)}\nINP")
                    colors.append('lightgreen')
                    positions.append(pos)
                    pos += 1
                
                # Add spacing between sizes
                pos += 0.5
            
            if len(plot_data) == 0:
                continue
            
            # Create boxplots
            bp = ax.boxplot(plot_data,
                           positions=positions,
                           widths=0.6,
                           patch_artist=True,
                           medianprops={'color': 'red', 'linewidth': 2},
                           whiskerprops={'color': 'black', 'linewidth': 1.5},
                           capprops={'color': 'black', 'linewidth': 1.5},
                           flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.5})
            
            # Color boxes
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            # Add mean markers
            for data, pos in zip(plot_data, positions):
                mean_val = np.mean(data)
                ax.plot(pos, mean_val, 'D', color='darkred', markersize=8, 
                       markeredgecolor='black', markeredgewidth=1, zorder=3)
            
            # Format axes
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('Kernel Time (µs)')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Use log scale if model suggests it
            if seg.get('model') == 'log-linear' or size_max / size_min > 1000:
                ax.set_yscale('log')
        
        # Overall title
        model_str = seg.get('model', 'unknown')
        r2_str = f"R²={seg.get('r_squared', 0):.3f}" if 'r_squared' in seg else ""
        fig.suptitle(
            f"{benchmark_name.upper()} - Segment {seg_num}: {format_size(size_min)} to {format_size(size_max)}\n"
            f"Model: {model_str}  {r2_str}",
            fontsize=14, fontweight='bold'
        )
        
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(output_dir, f'{benchmark_name}_segment{seg_num}_boxplots.png')
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        output_files.append(output_file)
        print(f"  Saved: {os.path.basename(output_file)}")
    
    return output_files


def print_statistics_summary(timing_df):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("TIMING STATISTICS SUMMARY")
    print("="*80)
    
    # Convert to microseconds
    timing_df['time_us'] = timing_df['time_seconds'] * 1e6
    
    # Overall statistics
    print("\nOverall Statistics:")
    print(f"  Total measurements: {len(timing_df)}")
    print(f"  Mean time: {timing_df['time_us'].mean():.2f} µs")
    print(f"  Std dev: {timing_df['time_us'].std():.2f} µs")
    print(f"  Min time: {timing_df['time_us'].min():.2f} µs")
    print(f"  Max time: {timing_df['time_us'].max():.2f} µs")
    
    # By operation mode
    print("\nBy Operation Mode:")
    for inplace_val in sorted(timing_df['inplace'].unique()):
        mode_name = "In-place" if inplace_val else "Out-of-place"
        mode_data = timing_df[timing_df['inplace'] == inplace_val]['time_us']
        print(f"  {mode_name}:")
        print(f"    Mean: {mode_data.mean():.2f} µs")
        print(f"    Measurements: {len(mode_data)}")
    
    # By size (show first and last few)
    print("\nSize Range:")
    sizes = sorted(timing_df['size_bytes'].unique())
    print(f"  Smallest: {format_size(sizes[0])}")
    print(f"  Largest: {format_size(sizes[-1])}")
    print(f"  Number of sizes: {len(sizes)}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Create boxplot visualizations of RCCL kernel timing distributions')
    parser.add_argument('run_dir',
                        help='Run directory containing timing data (e.g., run_all_reduce_20251104_165031)')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.run_dir):
        print(f"Error: Directory not found: {args.run_dir}")
        return 1
    
    # Extract benchmark name from directory
    dir_name = os.path.basename(args.run_dir.rstrip('/'))
    # Format: run_<benchmark>_YYYYMMDD_HHMMSS
    match = re.match(r'run_([a-z_]+)_\d{8}_\d{6}', dir_name)
    if match:
        benchmark_name = match.group(1)
    else:
        print(f"Warning: Could not extract benchmark name from directory: {dir_name}")
        benchmark_name = 'benchmark'
    
    print(f"Creating boxplots for: {benchmark_name}")
    print(f"Run directory: {args.run_dir}")
    
    # Load timing data
    timing_df = load_timing_data(args.run_dir)
    if timing_df is None or len(timing_df) == 0:
        print("Error: No timing data found")
        return 1
    
    print(f"  Loaded {len(timing_df)} timing measurements")
    
    # Load segmentation
    segmentation = load_bic_segmentation(args.run_dir, benchmark_name)
    if segmentation:
        print(f"  Loaded BIC segmentation: {segmentation['n_segments']} segments")
    else:
        print("  No BIC segmentation found, creating single boxplot")
    
    # Print statistics
    print_statistics_summary(timing_df)
    
    # Create boxplots
    print("\nCreating boxplot visualizations...")
    output_files = create_segment_boxplots(timing_df, segmentation, benchmark_name, args.run_dir)
    
    if output_files:
        print(f"\n✅ Generated {len(output_files)} boxplot visualization(s)")
        for f in output_files:
            print(f"   {os.path.basename(f)}")
        return 0
    else:
        print("Error: No boxplots were generated")
        return 1


if __name__ == "__main__":
    sys.exit(main())
