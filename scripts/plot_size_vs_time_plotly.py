#!/usr/bin/env python3
"""
Create interactive Plotly visualization of RCCL benchmark performance.

Features:
- Size vs. Time plot with log-log scale
- Separate traces for Out-of-Place (OOP) and In-Place (INP)
- Kernel timing mean with IQR (25-75 percentile) error bands
- Wall clock times from benchmark output
- BIC segmentation boundaries as vertical lines
- Interactive hover information

Usage:
    python plot_size_vs_time_plotly.py <run_directory>
"""

import os
import sys
import re
import glob
import json
import argparse
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_timing_data(run_dir):
    """Load all_rank*.csv files and combine them."""
    timing_files = glob.glob(os.path.join(run_dir, 'all_rank*.csv'))
    
    if not timing_files:
        return None
    
    dfs = []
    for f in timing_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
    
    if not dfs:
        return None
    
    combined = pd.concat(dfs, ignore_index=True)
    return combined


def parse_benchmark_output(output_file):
    """
    Parse benchmark output to extract wall clock times.
    
    Format:
    #       size         count      type   redop     root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
    #                                                         (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
              8              2     float     sum       -1    23.29    0.00    0.00      0    23.26    0.00    0.00      0|N/A
    
    Columns 6 and 10 are out-of-place and in-place times respectively.
    """
    wall_times = []
    
    with open(output_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Match data lines
            match = re.match(
                r'^\s*(\d+)\s+(\d+)\s+(\w+)\s+(\w+)\s+(-?\d+)\s+'
                r'([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+'
                r'([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+((\d+)|N/A)',
                line
            )
            
            if match:
                size_bytes = int(match.group(1))
                wall_time_oop_us = float(match.group(6))
                wall_time_inp_us = float(match.group(10))
                
                # Out-of-place
                wall_times.append({
                    'size_bytes': size_bytes,
                    'inplace': 0,
                    'wall_time_us': wall_time_oop_us
                })
                
                # In-place
                wall_times.append({
                    'size_bytes': size_bytes,
                    'inplace': 1,
                    'wall_time_us': wall_time_inp_us
                })
    
    return pd.DataFrame(wall_times)


def build_kernel_summary(timing_df, inplace_mode):
    """
    Build summary statistics for kernel timings.
    
    Args:
        timing_df: DataFrame with columns [size_bytes, inplace, iteration, time_seconds]
        inplace_mode: 0 for out-of-place, 1 for in-place
    
    Returns:
        DataFrame with columns [size_bytes, kernel_mean_us, kernel_std_us, 
                                kernel_min_us, kernel_max_us, p25_us, p75_us, count]
    """
    # Filter by inplace mode
    filtered = timing_df[timing_df['inplace'] == inplace_mode].copy()
    
    if len(filtered) == 0:
        return pd.DataFrame()
    
    # Convert to microseconds
    filtered['time_us'] = filtered['time_seconds'] * 1e6
    
    # Group by size and calculate statistics
    summary = filtered.groupby('size_bytes')['time_us'].agg([
        ('kernel_mean_us', 'mean'),
        ('kernel_std_us', 'std'),
        ('kernel_min_us', 'min'),
        ('kernel_max_us', 'max'),
        ('p25_us', lambda x: np.percentile(x, 25)),
        ('p75_us', lambda x: np.percentile(x, 75)),
        ('count', 'count')
    ]).reset_index()
    
    return summary


def load_bic_segmentation(run_dir, benchmark_name):
    """Load BIC segmentation JSON file."""
    seg_file = os.path.join(run_dir, f'{benchmark_name}_bic_segmentation.json')
    
    if not os.path.exists(seg_file):
        return None
    
    with open(seg_file, 'r') as f:
        return json.load(f)


def format_size(size_bytes):
    """Format size in human-readable form."""
    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.1f} GiB"
    elif size_bytes >= 1024**2:
        return f"{size_bytes / (1024**2):.1f} MiB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KiB"
    else:
        return f"{size_bytes} B"


def plot_interactive(run_dir, benchmark_name, timing_df, wall_df, segmentation):
    """Create interactive Plotly visualization."""
    
    # Build kernel summaries for OOP and INP
    kernel_df_oop = build_kernel_summary(timing_df, inplace_mode=0)
    kernel_df_inp = build_kernel_summary(timing_df, inplace_mode=1)
    
    # Filter wall clock data
    wall_df_oop = wall_df[wall_df['inplace'] == 0].copy()
    wall_df_inp = wall_df[wall_df['inplace'] == 1].copy()
    
    # Create figure
    fig = go.Figure()
    
    # Color scheme
    color_oop = 'rgb(31, 119, 180)'  # Blue
    color_inp = 'rgb(255, 127, 14)'  # Orange
    
    # Plot OOP kernel mean with IQR error bands
    if len(kernel_df_oop) > 0:
        x_oop = kernel_df_oop['size_bytes'].values
        y_mean_oop = kernel_df_oop['kernel_mean_us'].values
        y_p25_oop = kernel_df_oop['p25_us'].values
        y_p75_oop = kernel_df_oop['p75_us'].values
        
        # IQR band
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_oop, x_oop[::-1]]),
            y=np.concatenate([y_p75_oop, y_p25_oop[::-1]]),
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=True,
            name='OOP IQR (25-75%)',
            hoverinfo='skip'
        ))
        
        # Mean line
        fig.add_trace(go.Scatter(
            x=x_oop,
            y=y_mean_oop,
            mode='lines+markers',
            name='OOP Kernel Mean',
            line=dict(color=color_oop, width=2),
            marker=dict(size=6),
            error_y=dict(
                type='data',
                symmetric=False,
                array=y_p75_oop - y_mean_oop,
                arrayminus=y_mean_oop - y_p25_oop,
                visible=True,
                color=color_oop,
                thickness=1.5,
                width=4
            ),
            hovertemplate=(
                '<b>OOP Kernel</b><br>' +
                'Size: %{x} bytes<br>' +
                'Mean: %{y:.2f} µs<br>' +
                '<extra></extra>'
            )
        ))
    
    # Plot INP kernel mean with IQR error bands
    if len(kernel_df_inp) > 0:
        x_inp = kernel_df_inp['size_bytes'].values
        y_mean_inp = kernel_df_inp['kernel_mean_us'].values
        y_p25_inp = kernel_df_inp['p25_us'].values
        y_p75_inp = kernel_df_inp['p75_us'].values
        
        # IQR band
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_inp, x_inp[::-1]]),
            y=np.concatenate([y_p75_inp, y_p25_inp[::-1]]),
            fill='toself',
            fillcolor='rgba(255, 127, 14, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=True,
            name='INP IQR (25-75%)',
            hoverinfo='skip'
        ))
        
        # Mean line
        fig.add_trace(go.Scatter(
            x=x_inp,
            y=y_mean_inp,
            mode='lines+markers',
            name='INP Kernel Mean',
            line=dict(color=color_inp, width=2),
            marker=dict(size=6),
            error_y=dict(
                type='data',
                symmetric=False,
                array=y_p75_inp - y_mean_inp,
                arrayminus=y_mean_inp - y_p25_inp,
                visible=True,
                color=color_inp,
                thickness=1.5,
                width=4
            ),
            hovertemplate=(
                '<b>INP Kernel</b><br>' +
                'Size: %{x} bytes<br>' +
                'Mean: %{y:.2f} µs<br>' +
                '<extra></extra>'
            )
        ))
    
    # Plot OOP wall clock times
    if len(wall_df_oop) > 0:
        fig.add_trace(go.Scatter(
            x=wall_df_oop['size_bytes'],
            y=wall_df_oop['wall_time_us'],
            mode='markers',
            name='OOP Wall Clock',
            marker=dict(
                color=color_oop,
                size=8,
                symbol='circle-open',
                line=dict(width=2)
            ),
            hovertemplate=(
                '<b>OOP Wall Clock</b><br>' +
                'Size: %{x} bytes<br>' +
                'Time: %{y:.2f} µs<br>' +
                '<extra></extra>'
            )
        ))
    
    # Plot INP wall clock times
    if len(wall_df_inp) > 0:
        fig.add_trace(go.Scatter(
            x=wall_df_inp['size_bytes'],
            y=wall_df_inp['wall_time_us'],
            mode='markers',
            name='INP Wall Clock',
            marker=dict(
                color=color_inp,
                size=8,
                symbol='square-open',
                line=dict(width=2)
            ),
            hovertemplate=(
                '<b>INP Wall Clock</b><br>' +
                'Size: %{x} bytes<br>' +
                'Time: %{y:.2f} µs<br>' +
                '<extra></extra>'
            )
        ))
    
    # Add BIC segmentation lines
    if segmentation and 'breakpoint_sizes' in segmentation:
        for i, breakpoint in enumerate(segmentation['breakpoint_sizes']):
            fig.add_vline(
                x=breakpoint,
                line=dict(color='red', width=2, dash='dash'),
                annotation=dict(
                    text=f"Segment {i+1}",
                    textangle=-90,
                    yref='paper',
                    y=0.98,
                    showarrow=False,
                    font=dict(size=10, color='red')
                )
            )
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'{benchmark_name.upper()} Performance - Size vs. Time',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis=dict(
            title='Message Size (bytes)',
            type='log',
            gridcolor='lightgray',
            showgrid=True
        ),
        yaxis=dict(
            title='Time (µs)',
            type='log',
            gridcolor='lightgray',
            showgrid=True
        ),
        hovermode='closest',
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        plot_bgcolor='white',
        width=1200,
        height=700
    )
    
    # Save to HTML
    output_file = os.path.join(run_dir, f'{benchmark_name}_size_vs_time.html')
    fig.write_html(output_file)
    
    print(f"Saved interactive plot: {output_file}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Create interactive Plotly visualization of RCCL benchmark performance')
    parser.add_argument('run_dir', help='Run directory containing timing data')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.run_dir):
        print(f"Error: Directory not found: {args.run_dir}")
        return 1
    
    # Extract benchmark name from directory
    dir_name = os.path.basename(args.run_dir.rstrip('/'))
    # Format: run_<benchmark>_YYYYMMDD_HHMMSS
    if dir_name.startswith('run_'):
        parts = dir_name.split('_')
        if len(parts) >= 4:
            benchmark_name = '_'.join(parts[1:-2])
        else:
            benchmark_name = 'benchmark'
    else:
        benchmark_name = 'benchmark'
    
    print(f"Creating Plotly visualization for: {benchmark_name}")
    print(f"Run directory: {args.run_dir}")
    
    # Load timing data
    timing_df = load_timing_data(args.run_dir)
    if timing_df is None or len(timing_df) == 0:
        print("Error: No timing data found")
        return 1
    
    print(f"  Loaded {len(timing_df)} timing measurements")
    
    # Load benchmark output
    output_file = os.path.join(args.run_dir, f'{benchmark_name}_benchmark_output.txt')
    if not os.path.exists(output_file):
        print(f"Warning: Benchmark output not found: {output_file}")
        wall_df = pd.DataFrame()
    else:
        wall_df = parse_benchmark_output(output_file)
        print(f"  Loaded {len(wall_df)} wall clock measurements")
    
    # Load BIC segmentation
    segmentation = load_bic_segmentation(args.run_dir, benchmark_name)
    if segmentation:
        print(f"  Loaded BIC segmentation: {segmentation['n_segments']} segments")
    else:
        print("  Warning: No BIC segmentation found")
    
    # Create plot
    output_file = plot_interactive(args.run_dir, benchmark_name, timing_df, wall_df, segmentation)
    
    print(f"\n✅ Visualization complete!")
    print(f"Open in browser: file://{os.path.abspath(output_file)}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

