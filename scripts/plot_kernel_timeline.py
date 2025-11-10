#!/usr/bin/env python3
"""
Plot interactive kernel timeline showing kernel execution across MPI ranks.

Creates separate timeline visualizations per performance segment with:
- One row per MPI rank
- Horizontal bars for kernel execution (start to end time)
- Vertical lines marking benchmark Tstart/Tend timestamps
- Interactive hover information
- Zoomable/pannable interface

Usage:
    python plot_kernel_timeline.py <run_directory>
"""

import os
import sys
import json
import csv
import argparse
import socket
import time
from collections import defaultdict

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_timestamp_file(filepath):
    """
    Parse benchmark timestamp file.
    
    Returns list of dicts with timestamp events.
    """
    events = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('Tstart'):
                # Tstart 0: 123456789: all_reduce_size_1024_oop_type_float_op_sum_root_0
                parts = line.split(':', 2)
                if len(parts) >= 3:
                    rank_part = parts[0].replace('Tstart', '').strip()
                    timestamp = int(parts[1].strip())
                    config = parts[2].strip()
                    
                    events.append({
                        'rank': int(rank_part),
                        'timestamp_ns': timestamp,
                        'event_type': 'start',
                        'config': config
                    })
            
            elif line.startswith('Tend'):
                # Tend 0: 123456789
                parts = line.split(':')
                if len(parts) >= 2:
                    rank_part = parts[0].replace('Tend', '').strip()
                    timestamp = int(parts[1].strip())
                    
                    events.append({
                        'rank': int(rank_part),
                        'timestamp_ns': timestamp,
                        'event_type': 'end',
                        'config': None
                    })
    
    return events


def load_all_kernel_data(run_dir, rank_pid_map):
    """
    Load all kernel traces from all ranks into a single dataframe.
    
    Returns: DataFrame with columns: rank, pid, kernel_name, begin_ns, end_ns, duration_ns, is_nccl
    """
    hostname = socket.gethostname()
    rocprof_dir = os.path.join(run_dir, 'rocp', hostname)
    
    if not os.path.exists(rocprof_dir):
        print(f"  Error: ROCProfiler directory not found: {rocprof_dir}")
        return pd.DataFrame()
    
    kernel_dfs = []
    
    for rank, pid in sorted(rank_pid_map.items()):
        kernel_file = os.path.join(rocprof_dir, f'{pid}_kernel_trace.csv')
        if not os.path.exists(kernel_file):
            print(f"    Warning: Kernel trace not found for rank {rank} (PID {pid})")
            continue
        
        try:
            df = pd.read_csv(kernel_file)
            
            # Filter to kernel dispatches only
            df = df[df['Kind'] == 'KERNEL_DISPATCH'].copy()
            
            if df.empty:
                continue
            
            # Add rank and PID
            df['rank'] = rank
            df['pid'] = pid
            
            # Rename and select columns
            df = df.rename(columns={
                'Kernel_Name': 'kernel_name',
                'Start_Timestamp': 'begin_ns',
                'End_Timestamp': 'end_ns'
            })
            
            # Compute duration and NCCL flag
            df['duration_ns'] = df['end_ns'] - df['begin_ns']
            df['is_nccl'] = df['kernel_name'].str.contains('nccl', case=False, na=False)
            
            # Select final columns
            kernel_dfs.append(df[['rank', 'pid', 'kernel_name', 'begin_ns', 
                                   'end_ns', 'duration_ns', 'is_nccl']])
        except Exception as e:
            print(f"    Warning: Failed to load kernel trace for rank {rank}: {e}")
            continue
    
    if not kernel_dfs:
        return pd.DataFrame()
    
    return pd.concat(kernel_dfs, ignore_index=True)


def extract_size_from_config(config):
    """Extract message size from config string."""
    if not config:
        return 0
    parts = config.split('_')
    for i, part in enumerate(parts):
        if part == 'size' and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return 0


def load_timestamp_ranges(run_dir, ranks):
    """
    Load all timestamp events and pair them into ranges.
    
    Returns: DataFrame with columns: rank, size_bytes, operation_mode, start_ns, end_ns, duration_ns, config
    """
    range_records = []
    
    for rank in ranks:
        timestamp_file = os.path.join(run_dir, f'rank_{rank}_timestamps.txt')
        if not os.path.exists(timestamp_file):
            print(f"    Warning: Timestamp file not found for rank {rank}")
            continue
        
        events = parse_timestamp_file(timestamp_file)
        
        # Pair up Tstart/Tend events
        i = 0
        while i < len(events) - 1:
            if events[i]['event_type'] == 'start' and events[i+1]['event_type'] == 'end':
                config = events[i]['config']
                size = extract_size_from_config(config)
                mode = 'inp' if '_inp_' in config else 'oop'
                
                range_records.append({
                    'rank': rank,
                    'size_bytes': size,
                    'operation_mode': mode,
                    'start_ns': events[i]['timestamp_ns'],
                    'end_ns': events[i+1]['timestamp_ns'],
                    'config': config
                })
                i += 2
            else:
                i += 1
    
    if not range_records:
        return pd.DataFrame()
    
    df_ranges = pd.DataFrame(range_records)
    df_ranges['duration_ns'] = df_ranges['end_ns'] - df_ranges['start_ns']
    return df_ranges


def load_segmentation(run_dir):
    """Load BIC segmentation data if available."""
    # Find segmentation JSON file
    import glob
    pattern = os.path.join(run_dir, '*_bic_segmentation.json')
    seg_files = glob.glob(pattern)

    if not seg_files:
        print(f"  Warning: No segmentation file found in {run_dir}")
        return None

    seg_file = seg_files[0]
    print(f"  Loading segmentation from: {os.path.basename(seg_file)}")

    with open(seg_file, 'r') as f:
        segmentation = json.load(f)

    return segmentation


def filter_kernels_by_size_range(df_kernels, df_ranges, min_size, max_size):
    """
    Filter kernels to those within timestamp ranges for the specified size range.
    
    Returns: DataFrame with kernels that fall within the size range timestamp bounds
    """
    # Filter ranges by size
    df_ranges_filtered = df_ranges[
        (df_ranges['size_bytes'] >= min_size) & 
        (df_ranges['size_bytes'] <= max_size)
    ].copy()
    
    if df_ranges_filtered.empty:
        return pd.DataFrame()
    
    # Filter to NCCL kernels only
    df_nccl = df_kernels[df_kernels['is_nccl']].copy()
    
    if df_nccl.empty:
        return pd.DataFrame()
    
    # For each kernel, find if it falls within any filtered timestamp range
    results = []
    
    for rank in df_ranges_filtered['rank'].unique():
        df_rank_kernels = df_nccl[df_nccl['rank'] == rank]
        df_rank_ranges = df_ranges_filtered[df_ranges_filtered['rank'] == rank]
        
        for _, rng in df_rank_ranges.iterrows():
            mask = (
                (df_rank_kernels['begin_ns'] >= rng['start_ns']) &
                (df_rank_kernels['begin_ns'] <= rng['end_ns'])
            )
            df_match = df_rank_kernels[mask].copy()
            if not df_match.empty:
                df_match['size_bytes'] = rng['size_bytes']
                df_match['operation_mode'] = rng['operation_mode']
                df_match['range_start_ns'] = rng['start_ns']
                df_match['range_end_ns'] = rng['end_ns']
                results.append(df_match)
    
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def plot_kernel_timeline_size_range(run_dir, df_kernels, df_ranges, num_ranks, min_size, max_size, operation_mode=None):
    """
    Create interactive kernel timeline plot for a range of message sizes.

    Args:
        operation_mode: Optional filter for 'inp' or 'oop'. If None, includes both.

    Returns: (output_file, creation_time_seconds, file_size_bytes)
    """
    mode_str = f" ({operation_mode})" if operation_mode else ""
    print(f"\n  Creating plot for size range: {min_size} - {max_size} bytes{mode_str}")

    start_time = time.time()

    # Filter kernels using dataframe operations
    df_plot = filter_kernels_by_size_range(df_kernels, df_ranges, min_size, max_size)

    if df_plot.empty:
        print(f"    Warning: No data found in size range {min_size} - {max_size}")
        return None, 0, 0

    # Further filter by operation mode if specified
    if operation_mode:
        df_plot = df_plot[df_plot['operation_mode'] == operation_mode].copy()
        if df_plot.empty:
            print(f"    Warning: No {operation_mode} data found in size range {min_size} - {max_size}")
            return None, 0, 0

    # Get statistics
    sizes_found = sorted(df_plot['size_bytes'].unique())
    modes_found = sorted(df_plot['operation_mode'].unique())
    t_min = df_plot['range_start_ns'].min()
    t_max = df_plot['range_end_ns'].max()
    ranks_with_data = sorted(df_plot['rank'].unique())
    
    # Count total benchmark runs (number of unique timestamp ranges)
    df_ranges_used = df_ranges[
        (df_ranges['size_bytes'] >= min_size) &
        (df_ranges['size_bytes'] <= max_size)
    ]
    if operation_mode:
        df_ranges_used = df_ranges_used[df_ranges_used['operation_mode'] == operation_mode]
    
    total_runs = len(df_ranges_used)

    print(f"    Sizes found: {sizes_found}")
    print(f"    Number of sizes: {len(sizes_found)}")
    print(f"    Operation modes: {modes_found}")
    print(f"    Time range: {(t_max - t_min) / 1e9:.2f} seconds")
    print(f"    Ranks with data: {len(ranks_with_data)}")
    print(f"    Benchmark runs: {total_runs}")
    print(f"    Total kernels: {len(df_plot)}")

    # Create figure with subplots (one per rank)
    fig = make_subplots(
        rows=num_ranks,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01  # Tighter spacing between subplots
    )

    # Color schemes
    colors = {
        'nccl_kernel': 'rgb(31, 119, 180)',      # Blue for NCCL kernels
        'other_kernel': 'rgb(200, 200, 200)',    # Gray for other kernels
        'tstart': 'rgb(0, 200, 0)',              # Bright green for Tstart (fully opaque)
        'tend': 'rgb(255, 0, 0)',                # Bright red for Tend (fully opaque)
        'interval_bar': 'rgba(255, 165, 0, 0.3)', # Orange for interval span
    }

    # Track whether we've added legend entries (only add once)
    interval_legend_added = False

    # Plot each rank
    for row_idx in range(1, num_ranks + 1):
        rank = row_idx - 1

        # Get data for this rank
        df_rank_kernels = df_plot[df_plot['rank'] == rank]
        df_rank_ranges = df_ranges_used[df_ranges_used['rank'] == rank]

        if df_rank_kernels.empty:
            continue

        # Plot kernels as horizontal bars
        for _, kernel in df_rank_kernels.iterrows():
            start_rel = (kernel['begin_ns'] - t_min) / 1e6  # Convert to milliseconds
            end_rel = (kernel['end_ns'] - t_min) / 1e6
            duration_us = kernel['duration_ns'] / 1000.0

            # Shorten kernel name for display
            kernel_short = kernel['kernel_name'].split('(')[0]
            if len(kernel_short) > 40:
                kernel_short = kernel_short[:37] + '...'

            fig.add_trace(
                go.Scatter(
                    x=[start_rel, end_rel],
                    y=[rank, rank],
                    mode='lines',
                    line=dict(color=colors['nccl_kernel'], width=8),
                    hovertemplate=(
                        f"<b>Rank {rank}</b><br>" +
                        f"Kernel: {kernel_short}<br>" +
                        f"Start: {start_rel:.2f} ms<br>" +
                        f"End: {end_rel:.2f} ms<br>" +
                        f"Duration: {duration_us:.2f} μs<br>" +
                        f"Size: {kernel['size_bytes']} bytes<br>" +
                        "<extra></extra>"
                    ),
                    showlegend=False,
                    name=f"Rank {rank} Kernel"
                ),
                row=row_idx,
                col=1
            )

        # Plot timestamp markers and interval spans
        if not df_rank_ranges.empty:
            tstart_times = []
            tend_times = []
            tstart_configs = []
            tend_configs = []

            for _, rng in df_rank_ranges.iterrows():
                time_start_rel = (rng['start_ns'] - t_min) / 1e6  # ms
                time_end_rel = (rng['end_ns'] - t_min) / 1e6  # ms
                interval_duration_ms = time_end_rel - time_start_rel

                # Calculate kernel time for this specific range
                range_kernels = df_plot[
                    (df_plot['rank'] == rank) &
                    (df_plot['range_start_ns'] == rng['start_ns']) &
                    (df_plot['range_end_ns'] == rng['end_ns'])
                ]
                kernel_time_ms = range_kernels['duration_ns'].sum() / 1e6
                gap_time_ms = interval_duration_ms - kernel_time_ms
                gap_pct = (gap_time_ms / interval_duration_ms * 100) if interval_duration_ms > 0 else 0

                tstart_times.append(time_start_rel)
                tstart_configs.append(f"Size: {rng['size_bytes']} bytes ({rng['operation_mode']})")

                tend_times.append(time_end_rel)
                tend_configs.append(f"Duration: {interval_duration_ms:.2f} ms")
                
                # Add a horizontal bar spanning the interval to show total time
                # Position it at rank + 0.4 (above the kernel traces)
                fig.add_trace(
                    go.Scatter(
                        x=[time_start_rel, time_end_rel],
                        y=[rank + 0.4, rank + 0.4],
                        mode='lines',
                        line=dict(color='rgba(255, 165, 0, 0.8)', width=15),
                        fill=None,
                        hovertemplate=(
                            f"<b>Rank {rank} - Timestamp Interval</b><br>" +
                            f"Size: {rng['size_bytes']} bytes ({rng['operation_mode']})<br>" +
                            f"Start: {time_start_rel:.2f} ms<br>" +
                            f"End: {time_end_rel:.2f} ms<br>" +
                            f"<b>Interval Duration: {interval_duration_ms:.2f} ms</b><br>" +
                            f"Kernel Time: {kernel_time_ms:.2f} ms ({100-gap_pct:.1f}%)<br>" +
                            f"Launch Gaps: {gap_time_ms:.2f} ms ({gap_pct:.1f}%)<br>" +
                            "<extra></extra>"
                        ),
                        showlegend=not interval_legend_added,
                        name="Timestamp Interval",
                        legendgroup="interval"
                    ),
                    row=row_idx,
                    col=1
                )
                interval_legend_added = True

            # Add scatter points for Tstart/Tend hover info (below the kernel line, pointing up)
            if tstart_times:
                fig.add_trace(
                    go.Scatter(
                        x=tstart_times,
                        y=[rank - 0.25] * len(tstart_times),  # Below the kernel line
                        mode='markers',
                        marker=dict(size=8, color=colors['tstart'], symbol='triangle-up',
                                   line=dict(width=1, color='darkgreen')),
                        hovertemplate=(
                            f"<b>Rank {rank} - Tstart</b><br>" +
                            "Time: %{x:.2f} ms<br>" +
                            "%{text}<br>" +
                            "<extra></extra>"
                        ),
                        text=tstart_configs,
                        showlegend=(row_idx == 1),
                        name="Benchmark Start",
                        legendgroup="tstart"
                    ),
                    row=row_idx,
                    col=1
                )

            if tend_times:
                fig.add_trace(
                    go.Scatter(
                        x=tend_times,
                        y=[rank - 0.25] * len(tend_times),  # Below the kernel line
                        mode='markers',
                        marker=dict(size=8, color=colors['tend'], symbol='triangle-up',
                                   line=dict(width=1, color='darkred')),
                        hovertemplate=(
                            f"<b>Rank {rank} - Tend</b><br>" +
                            "Time: %{x:.2f} ms<br>" +
                            "%{text}<br>" +
                            "<extra></extra>"
                        ),
                        text=tend_configs,
                        showlegend=(row_idx == 1),
                        name="Benchmark End",
                        legendgroup="tend"
                    ),
                    row=row_idx,
                    col=1
                )

    # Calculate size centerpoints for secondary X axis
    size_centers = {}
    for size in sizes_found:
        size_ranges = df_ranges_used[df_ranges_used['size_bytes'] == size]
        if not size_ranges.empty:
            # Calculate center point in milliseconds relative to t_min
            centers = []
            for _, rng in size_ranges.iterrows():
                start_rel = (rng['start_ns'] - t_min) / 1e6
                end_rel = (rng['end_ns'] - t_min) / 1e6
                centers.append((start_rel + end_rel) / 2)
            size_centers[size] = sum(centers) / len(centers)
    
    # Add timing analysis annotations
    # Calculate timing statistics properly matched by range
    num_ranges = len(df_ranges_used)
    
    # For each timestamp range, calculate kernel time
    range_stats = []
    for _, rng in df_ranges_used.iterrows():
        interval_time_ms = rng['duration_ns'] / 1e6
        
        # Find kernels for this specific range
        range_kernels = df_plot[
            (df_plot['rank'] == rng['rank']) &
            (df_plot['range_start_ns'] == rng['start_ns']) &
            (df_plot['range_end_ns'] == rng['end_ns'])
        ]
        kernel_time_ms = range_kernels['duration_ns'].sum() / 1e6
        gap_time_ms = interval_time_ms - kernel_time_ms
        
        range_stats.append({
            'interval_ms': interval_time_ms,
            'kernel_ms': kernel_time_ms,
            'gap_ms': gap_time_ms
        })
    
    if range_stats:
        avg_interval_time_ms = sum(s['interval_ms'] for s in range_stats) / len(range_stats)
        avg_kernel_time_ms = sum(s['kernel_ms'] for s in range_stats) / len(range_stats)
        avg_gap_time_ms = sum(s['gap_ms'] for s in range_stats) / len(range_stats)
        avg_gap_percent = (avg_gap_time_ms / avg_interval_time_ms * 100) if avg_interval_time_ms > 0 else 0
        
        # Create single two-column timing analysis using manual spacing
        # Using monospace font to ensure alignment
        annotation_text = (
            f"<b>Timing Analysis (Avg per Range)</b><br>"
            f"Num Ranges: {num_ranges:3d}     Config: {len(sizes_found)} sizes × {num_ranges//len(sizes_found)} ranks<br>"
            f"Avg Interval: {avg_interval_time_ms:6.2f} ms     Kernel: {avg_kernel_time_ms:.2f} ms ({100-avg_gap_percent:.1f}%)<br>"
            f"                                   Gaps: {avg_gap_time_ms:.2f} ms ({avg_gap_percent:.1f}%)<br>"
            f"<i>Hover orange bars for per-range details</i>"
        )
    else:
        annotation_text = "<b>Timing Analysis:</b><br>No data available"
    
    # Update layout
    dirpath = run_dir.rstrip('/ \t\n')
    benchmark_name = os.path.basename(dirpath).split('run_')[1].rsplit('_', 2)[0]
    
    # Format size range for display
    def format_size(size_bytes):
        if size_bytes >= 1024*1024*1024:
            return f"{size_bytes / (1024*1024*1024):.1f} GiB"
        elif size_bytes >= 1024*1024:
            return f"{size_bytes / (1024*1024):.1f} MiB"
        elif size_bytes >= 1024:
            return f"{size_bytes / 1024:.1f} KiB"
        else:
            return f"{size_bytes} B"
    
    min_str = format_size(min_size)
    max_str = format_size(max_size)
    
    # Add operation mode to title
    mode_label = ""
    if operation_mode == 'inp':
        mode_label = " - In-Place"
    elif operation_mode == 'oop':
        mode_label = " - Out-of-Place"
    elif len(modes_found) == 1:
        mode_label = f" - {'In-Place' if modes_found[0] == 'inp' else 'Out-of-Place'}"

    fig.update_layout(
        title={
            'text': f"RCCL Kernel Timeline - {benchmark_name}{mode_label}",
            'x': 0.5,
            'xanchor': 'center'
        },
        height=80 * num_ranks + 220,  # Very tight layout with extra space at top for annotation and labels
        margin=dict(t=150, b=50, l=80, r=50),  # Extra top margin to separate title from plot area
        hovermode='closest',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,  # Move legend higher to avoid overlapping size labels
            xanchor="right",
            x=1
        ),
        annotations=[]
    )

    # Add single timing analysis annotation (two columns) above the plot area
    fig.add_annotation(
        text=annotation_text,
        xref="paper",
        yref="paper",
        x=0.02,
        y=1.05,  # Above the plot area in the top margin
        xanchor="left",
        yanchor="bottom",
        showarrow=False,
        bgcolor="rgba(240, 248, 255, 0.95)",  # Light blue with high opacity for contrast
        bordercolor="rgba(0, 0, 0, 0)",  # No border (transparent)
        borderwidth=0,
        borderpad=6,
        font=dict(size=11, family="Courier New, monospace", color="black"),  # Larger font with explicit black color
        # Note: borderradius not supported in Plotly annotations
    )

    # Update axes
    # Bottom x-axis (time)
    fig.update_xaxes(
        title_text="Time (milliseconds from start)",
        row=num_ranks,
        col=1
    )
    
    # Add size labels as annotations above the plot area (must be y > 1.0 to stay in margin)
    if size_centers:
        # Add a "Message Size (bytes)" label
        fig.add_annotation(
            x=0.5,
            y=1.015,
            text="Message Size (bytes):",
            showarrow=False,
            xref='paper',
            yref='paper',
            xanchor='center',
            yanchor='bottom',
            font=dict(size=10, color='blue', family='Arial Black'),
            bgcolor='rgba(230, 240, 255, 0.9)',
            borderpad=2
        )
        # Add individual size labels below the "Message Size" title
        for size, center_pos in size_centers.items():
            fig.add_annotation(
                x=center_pos,
                y=1.0,
                text=f"{size} B",
                showarrow=False,
                xref='x1',
                yref='paper',
                xanchor='center',
                yanchor='bottom',
                font=dict(size=9, color='blue'),
                bgcolor='rgba(255, 255, 255, 0.8)',
                borderpad=1
            )

    for row_idx in range(1, num_ranks + 1):
        rank = row_idx - 1
        fig.update_yaxes(
            showticklabels=True,
            range=[rank - 0.3, rank + 0.7],  # Tight range around rank position
            tickmode='array',
            tickvals=[rank],
            ticktext=[f"Rank {rank}"],
            row=row_idx,
            col=1
        )

    # Validate layout for potential visual issues
    layout_issues = validate_layout_parameters(fig, num_ranks, size_centers)
    if layout_issues:
        print("  Layout validation warnings:")
        for issue in layout_issues:
            print(f"    ⚠️  {issue}")

    # Save to HTML
    mode_suffix = f"_{operation_mode}" if operation_mode else ""
    output_file = os.path.join(run_dir, f'kernel_timeline_range_{min_size}_{max_size}{mode_suffix}.html')
    fig.write_html(output_file)

    creation_time = time.time() - start_time
    file_size = os.path.getsize(output_file)

    print(f"    ✓ Created in {creation_time:.2f} seconds")
    print(f"    ✓ File size: {file_size / 1024 / 1024:.2f} MB")

    return output_file, creation_time, file_size


def plot_kernel_timeline(run_dir, min_size=None, max_size=None, mode='both'):
    """
    Create interactive kernel timeline plots.

    If min_size and max_size are provided, creates plots for that size range.
    Otherwise, creates one plot per segment from BIC segmentation.

    Args:
        mode: 'inp', 'oop', or 'both' - controls which operation modes to plot

    Returns: List of (output_file, creation_time, file_size) tuples
    """
    print(f"Creating kernel timeline for: {run_dir}")

    # Determine mode: size range or segmentation
    use_size_range = min_size is not None and max_size is not None

    if not use_size_range:
        # Load segmentation for segment-based plotting
        segmentation = load_segmentation(run_dir)
        if not segmentation:
            print("  Error: Segmentation data required (or specify --min-size and --max-size)")
            return []

        segments = segmentation.get('segments', [])
        if not segments:
            print("  Error: No segments found in segmentation data")
            return []

        print(f"  Found {len(segments)} segment(s)")
    else:
        print(f"  Using size range: {min_size} - {max_size} bytes")
        print(f"  Operation mode: {mode}")

    # Load rank-to-PID mapping
    rank_pid_file = os.path.join(run_dir, 'rank_pid_mapping.json')
    if not os.path.exists(rank_pid_file):
        print(f"  Error: No rank_pid_mapping.json found")
        return []

    with open(rank_pid_file, 'r') as f:
        rank_pid_map = json.load(f)

    rank_pid_map = {int(k): int(v) for k, v in rank_pid_map.items()}
    num_ranks = len(rank_pid_map)
    print(f"  Found {num_ranks} ranks")

    # Load all data into dataframes
    print("\n  Loading data into dataframes...")
    
    # Load all kernel traces
    df_kernels = load_all_kernel_data(run_dir, rank_pid_map)
    if df_kernels.empty:
        print("  Error: No kernel data loaded")
        return []
    
    print(f"    ✓ Loaded {len(df_kernels)} kernel dispatches")
    print(f"    ✓ {df_kernels['is_nccl'].sum()} NCCL kernels")
    
    # Load timestamp ranges
    df_ranges = load_timestamp_ranges(run_dir, sorted(rank_pid_map.keys()))
    if df_ranges.empty:
        print("  Error: No timestamp data loaded")
        return []
    
    print(f"    ✓ Loaded {len(df_ranges)} timestamp ranges")
    print(f"  ✓ Data loading complete")

    # Create plots based on mode
    results = []
    
    # Determine which operation modes to plot
    if mode == 'both':
        modes_to_plot = ['inp', 'oop']
    else:
        modes_to_plot = [mode]
    
    if use_size_range:
        # Create plots for specified size range
        for op_mode in modes_to_plot:
            result = plot_kernel_timeline_size_range(
                run_dir, df_kernels, df_ranges, num_ranks, min_size, max_size, op_mode
            )
            if result and result[0]:  # Check if output_file is not None
                results.append(result)
    else:
        # Create plots for each segment
        for segment in segments:
            seg_idx = segment['segment']
            seg_min = segment['size_range_bytes'][0]
            seg_max = segment['size_range_bytes'][1]
            
            for op_mode in modes_to_plot:
                result = plot_kernel_timeline_size_range(
                    run_dir, df_kernels, df_ranges, num_ranks, seg_min, seg_max, op_mode
                )
                if result and result[0]:  # Check if output_file is not None
                    # Rename output file for segment
                    old_file = result[0]
                    new_file = os.path.join(run_dir, f'kernel_timeline_segment{seg_idx}_{op_mode}.html')
                    os.rename(old_file, new_file)
                    results.append((new_file, result[1], result[2]))

    return results


def validate_layout_parameters(fig, num_ranks, size_centers):
    """
    Basic validation of layout parameters to detect potential visual issues.
    Note: Cannot detect actual visual overlaps without rendering the plot.
    """
    issues = []

    # Check if we have too many size labels for the available space
    if size_centers and len(size_centers) > 12:
        issues.append(f"Warning: {len(size_centers)} size labels may cause crowding")

    # Check for potential subplot height issues
    height = 80 * num_ranks + 220
    if height > 2000:
        issues.append(f"Warning: Plot height ({height}px) may be too tall for display")

    # Check for potential margin issues
    layout = fig.layout
    if hasattr(layout, 'margin') and layout.margin.t < 100:
        issues.append("Warning: Top margin may be too small for all labels")

    return issues


def main():
    parser = argparse.ArgumentParser(
        description='Create interactive kernel timeline visualizations',
        epilog="""
Examples:
  # Create plots for all segments (from BIC segmentation):
  %(prog)s /path/to/run_dir

  # Create plot for a specific size range:
  %(prog)s /path/to/run_dir --min-size 1024 --max-size 1048576
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('run_dir', help='Run directory containing timing data')
    parser.add_argument('--min-size', type=int, metavar='BYTES',
                        help='Minimum message size (bytes) for custom range')
    parser.add_argument('--max-size', type=int, metavar='BYTES',
                        help='Maximum message size (bytes) for custom range')
    parser.add_argument('--mode', choices=['inp', 'oop', 'both'], default='both',
                        help='Operation mode: inp (in-place), oop (out-of-place), or both (default: both)')

    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        print(f"Error: Directory not found: {args.run_dir}")
        return 1

    # Validate size range arguments
    if (args.min_size is not None) != (args.max_size is not None):
        print("Error: Both --min-size and --max-size must be specified together")
        return 1

    if args.min_size is not None and args.min_size > args.max_size:
        print("Error: --min-size must be less than or equal to --max-size")
        return 1

    results = plot_kernel_timeline(args.run_dir, args.min_size, args.max_size, args.mode)

    if results:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        total_time = sum(r[1] for r in results)
        total_size = sum(r[2] for r in results)

        for output_file, creation_time, file_size in results:
            seg_name = os.path.basename(output_file)
            print(f"\n{seg_name}:")
            print(f"  Creation time: {creation_time:.2f} seconds")
            print(f"  File size: {file_size / 1024 / 1024:.2f} MB")
            print(f"  Path: file://{os.path.abspath(output_file)}")

        print(f"\nTotal creation time: {total_time:.2f} seconds")
        print(f"Total file size: {total_size / 1024 / 1024:.2f} MB")
        print("=" * 80)

        return 0
    else:
        print("\n❌ No plots created")
        return 1


if __name__ == '__main__':
    sys.exit(main())

