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


def parse_rocprof_kernel_trace(filepath):
    """
    Parse ROCProfiler kernel trace CSV.
    
    Returns list of dicts with kernel execution data.
    """
    kernels = []
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Only process kernel dispatches
                if row.get('Kind') != 'KERNEL_DISPATCH':
                    continue
                
                kernel_name = row.get('Kernel_Name', '')
                start_ts = int(row.get('Start_Timestamp', 0))
                end_ts = int(row.get('End_Timestamp', 0))
                
                if start_ts == 0 or end_ts == 0:
                    continue
                
                kernel = {
                    'kernel_name': kernel_name,
                    'begin_ns': start_ts,
                    'end_ns': end_ts,
                    'duration_ns': end_ts - start_ts
                }
                
                kernels.append(kernel)
            except (ValueError, KeyError) as e:
                continue
    
    return kernels


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


def plot_kernel_timeline_size_range(run_dir, rank_data, num_ranks, min_size, max_size):
    """
    Create interactive kernel timeline plot for a range of message sizes.

    Returns: (output_file, creation_time_seconds, file_size_bytes)
    """
    print(f"\n  Creating plot for size range: {min_size} - {max_size} bytes")

    start_time = time.time()

    # Filter timestamp events to only include sizes in the range
    filtered_rank_data = {}
    all_timestamps = []
    sizes_found = set()

    for rank, data in rank_data.items():
        # Filter timestamp events by size range
        filtered_events = []
        for event in data['timestamp_events']:
            if event['event_type'] == 'start':
                size = extract_size_from_config(event['config'])
                if min_size <= size <= max_size:
                    filtered_events.append(event)
                    sizes_found.add(size)
            elif event['event_type'] == 'end':
                # Keep all 'end' events that follow a matching 'start'
                filtered_events.append(event)

        if not filtered_events:
            continue

        # Get time range for this size's events
        event_times = [e['timestamp_ns'] for e in filtered_events]
        if not event_times:
            continue

        t_min_seg = min(event_times)
        t_max_seg = max(event_times)

        # Filter kernels to only those within this exact time range
        filtered_kernels = [
            k for k in data['kernels']
            if t_min_seg <= k['begin_ns'] <= t_max_seg
        ]
        filtered_nccl_kernels = [
            k for k in data['nccl_kernels']
            if t_min_seg <= k['begin_ns'] <= t_max_seg
        ]

        filtered_rank_data[rank] = {
            'timestamp_events': filtered_events,
            'kernels': filtered_kernels,
            'nccl_kernels': filtered_nccl_kernels
        }

        all_timestamps.extend(event_times)

    if not filtered_rank_data:
        print(f"    Warning: No data found in size range {min_size} - {max_size}")
        return None, 0, 0

    # Find time range for this size range
    all_times = all_timestamps.copy()
    for rank_info in filtered_rank_data.values():
        for k in rank_info['kernels']:
            all_times.extend([k['begin_ns'], k['end_ns']])

    t_min = min(all_times)
    t_max = max(all_times)

    print(f"    Sizes found: {sorted(sizes_found)}")
    print(f"    Number of sizes: {len(sizes_found)}")
    print(f"    Time range: {(t_max - t_min) / 1e9:.2f} seconds")
    print(f"    Ranks with data: {len(filtered_rank_data)}")

    # Count total benchmark runs (Tstart events)
    total_runs = sum(
        len([e for e in data['timestamp_events'] if e['event_type'] == 'start'])
        for data in filtered_rank_data.values()
    )
    print(f"    Benchmark runs: {total_runs}")

    # Create figure with subplots (one per rank)
    fig = make_subplots(
        rows=num_ranks,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=[f"Rank {r}" for r in range(num_ranks)]
    )

    # Color schemes
    colors = {
        'nccl_kernel': 'rgb(31, 119, 180)',      # Blue for NCCL kernels
        'other_kernel': 'rgb(200, 200, 200)',    # Gray for other kernels
        'tstart': 'rgba(0, 255, 0, 0.5)',        # Green for Tstart
        'tend': 'rgba(255, 0, 0, 0.5)',          # Red for Tend
    }

    # Plot each rank
    for row_idx in range(1, num_ranks + 1):
        rank = row_idx - 1

        if rank not in filtered_rank_data:
            continue

        data = filtered_rank_data[rank]

        # Plot kernels as horizontal bars
        for kernel in data['nccl_kernels']:
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
                        "<extra></extra>"
                    ),
                    showlegend=False,
                    name=f"Rank {rank} Kernel"
                ),
                row=row_idx,
                col=1
            )

        # Plot timestamp markers
        tstart_times = []
        tend_times = []
        tstart_configs = []

        for event in data['timestamp_events']:
            time_rel = (event['timestamp_ns'] - t_min) / 1e6  # ms

            if event['event_type'] == 'start':
                tstart_times.append(time_rel)
                size = extract_size_from_config(event['config'])
                tstart_configs.append(f"Size: {size} bytes")

                # Add vertical line for Tstart
                fig.add_vline(
                    x=time_rel,
                    line=dict(color=colors['tstart'], width=1, dash='dash'),
                    row=row_idx,
                    col=1
                )
            elif event['event_type'] == 'end':
                tend_times.append(time_rel)

                # Add vertical line for Tend
                fig.add_vline(
                    x=time_rel,
                    line=dict(color=colors['tend'], width=1, dash='dash'),
                    row=row_idx,
                    col=1
                )

        # Add scatter points for Tstart/Tend hover info
        if tstart_times:
            fig.add_trace(
                go.Scatter(
                    x=tstart_times,
                    y=[rank] * len(tstart_times),
                    mode='markers',
                    marker=dict(size=8, color=colors['tstart'], symbol='triangle-up'),
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
                    y=[rank] * len(tend_times),
                    mode='markers',
                    marker=dict(size=8, color=colors['tend'], symbol='triangle-down'),
                    hovertemplate=(
                        f"<b>Rank {rank} - Tend</b><br>" +
                        "Time: %{x:.2f} ms<br>" +
                        "<extra></extra>"
                    ),
                    showlegend=(row_idx == 1),
                    name="Benchmark End",
                    legendgroup="tend"
                ),
                row=row_idx,
                col=1
            )

    # Update layout
    benchmark_name = os.path.basename(run_dir).split('run_')[1].rsplit('_', 2)[0]
    
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

    fig.update_layout(
        title={
            'text': f"RCCL Kernel Timeline - {benchmark_name}<br>"
                    f"<sub>Size range: {min_str} - {max_str} ({len(sizes_found)} sizes)</sub>",
            'x': 0.5,
            'xanchor': 'center'
        },
        height=200 * num_ranks + 100,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Update axes
    fig.update_xaxes(
        title_text="Time (milliseconds from start)",
        row=num_ranks,
        col=1
    )

    for row_idx in range(1, num_ranks + 1):
        fig.update_yaxes(
            showticklabels=False,
            row=row_idx,
            col=1
        )

    # Save to HTML
    output_file = os.path.join(run_dir, f'kernel_timeline_range_{min_size}_{max_size}.html')
    fig.write_html(output_file)

    creation_time = time.time() - start_time
    file_size = os.path.getsize(output_file)

    print(f"    ✓ Created in {creation_time:.2f} seconds")
    print(f"    ✓ File size: {file_size / 1024 / 1024:.2f} MB")

    return output_file, creation_time, file_size


def plot_kernel_timeline_segment(run_dir, rank_data, num_ranks, segment_info, segment_idx):
    """
    Create interactive kernel timeline plot for a single segment.

    Returns: (output_file, creation_time_seconds, file_size_bytes)
    """
    seg_min, seg_max = segment_info['size_range_bytes']
    print(f"\n  Creating plot for segment {segment_idx}")
    print(f"    Size range: {seg_min} - {seg_max} bytes")

    start_time = time.time()

    # Filter timestamp events to only include sizes in this segment
    filtered_rank_data = {}
    all_timestamps = []

    for rank, data in rank_data.items():
        # Filter timestamp events by size
        filtered_events = []
        for event in data['timestamp_events']:
            if event['event_type'] == 'start':
                size = extract_size_from_config(event['config'])
                if seg_min <= size <= seg_max:
                    filtered_events.append(event)
            elif event['event_type'] == 'end':
                # Keep all 'end' events that follow a matching 'start'
                # For simplicity, include all end events within reasonable proximity
                filtered_events.append(event)

        if not filtered_events:
            continue

        # Get time range for this segment's events
        event_times = [e['timestamp_ns'] for e in filtered_events]
        if not event_times:
            continue

        t_min_seg = min(event_times)
        t_max_seg = max(event_times)

        # Filter kernels to only those within this exact time range
        filtered_kernels = [
            k for k in data['kernels']
            if t_min_seg <= k['begin_ns'] <= t_max_seg
        ]
        filtered_nccl_kernels = [
            k for k in data['nccl_kernels']
            if t_min_seg <= k['begin_ns'] <= t_max_seg
        ]

        filtered_rank_data[rank] = {
            'timestamp_events': filtered_events,
            'kernels': filtered_kernels,
            'nccl_kernels': filtered_nccl_kernels
        }

        all_timestamps.extend(event_times)

    if not filtered_rank_data:
        print(f"    Warning: No data in this segment")
        return None, 0, 0

    # Find time range for this segment
    all_times = all_timestamps.copy()
    for rank_info in filtered_rank_data.values():
        for k in rank_info['kernels']:
            all_times.extend([k['begin_ns'], k['end_ns']])

    t_min = min(all_times)
    t_max = max(all_times)

    print(f"    Time range: {(t_max - t_min) / 1e9:.2f} seconds")
    print(f"    Ranks with data: {len(filtered_rank_data)}")

    # Create figure with subplots (one per rank)
    fig = make_subplots(
        rows=num_ranks,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=[f"Rank {r}" for r in range(num_ranks)]
    )

    # Color schemes
    colors = {
        'nccl_kernel': 'rgb(31, 119, 180)',      # Blue for NCCL kernels
        'other_kernel': 'rgb(200, 200, 200)',    # Gray for other kernels
        'tstart': 'rgba(0, 255, 0, 0.5)',        # Green for Tstart
        'tend': 'rgba(255, 0, 0, 0.5)',          # Red for Tend
    }

    # Plot each rank
    for row_idx in range(1, num_ranks + 1):
        rank = row_idx - 1

        if rank not in filtered_rank_data:
            # Add empty trace for ranks with no data
            continue

        data = filtered_rank_data[rank]

        # Plot kernels as horizontal bars
        for kernel in data['nccl_kernels']:
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
                        "<extra></extra>"
                    ),
                    showlegend=False,
                    name=f"Rank {rank} Kernel"
                ),
                row=row_idx,
                col=1
            )

        # Plot timestamp markers
        tstart_times = []
        tend_times = []
        tstart_configs = []

        for event in data['timestamp_events']:
            time_rel = (event['timestamp_ns'] - t_min) / 1e6  # ms

            if event['event_type'] == 'start':
                tstart_times.append(time_rel)
                size = extract_size_from_config(event['config'])
                tstart_configs.append(f"Size: {size} bytes")

                # Add vertical line for Tstart
                fig.add_vline(
                    x=time_rel,
                    line=dict(color=colors['tstart'], width=1, dash='dash'),
                    row=row_idx,
                    col=1
                )
            elif event['event_type'] == 'end':
                tend_times.append(time_rel)

                # Add vertical line for Tend
                fig.add_vline(
                    x=time_rel,
                    line=dict(color=colors['tend'], width=1, dash='dash'),
                    row=row_idx,
                    col=1
                )

        # Add invisible scatter points for Tstart/Tend hover info
        if tstart_times:
            fig.add_trace(
                go.Scatter(
                    x=tstart_times,
                    y=[rank] * len(tstart_times),
                    mode='markers',
                    marker=dict(size=8, color=colors['tstart'], symbol='triangle-up'),
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
                    y=[rank] * len(tend_times),
                    mode='markers',
                    marker=dict(size=8, color=colors['tend'], symbol='triangle-down'),
                    hovertemplate=(
                        f"<b>Rank {rank} - Tend</b><br>" +
                        "Time: %{x:.2f} ms<br>" +
                        "<extra></extra>"
                    ),
                    showlegend=(row_idx == 1),
                    name="Benchmark End",
                    legendgroup="tend"
                ),
                row=row_idx,
                col=1
            )

    # Update layout
    benchmark_name = os.path.basename(run_dir).split('run_')[1].rsplit('_', 2)[0]
    fig.update_layout(
        title={
            'text': f"RCCL Kernel Timeline - {benchmark_name} - Segment {segment_idx}<br>"
                    f"<sub>Size range: {seg_min:,} - {seg_max:,} bytes</sub>",
            'x': 0.5,
            'xanchor': 'center'
        },
        height=200 * num_ranks + 100,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Update axes
    fig.update_xaxes(
        title_text="Time (milliseconds from segment start)",
        row=num_ranks,
        col=1
    )

    for row_idx in range(1, num_ranks + 1):
        fig.update_yaxes(
            showticklabels=False,
            row=row_idx,
            col=1
        )

    # Save to HTML
    output_file = os.path.join(run_dir, f'kernel_timeline_segment{segment_idx}.html')
    fig.write_html(output_file)

    creation_time = time.time() - start_time
    file_size = os.path.getsize(output_file)

    print(f"    ✓ Created in {creation_time:.2f} seconds")
    print(f"    ✓ File size: {file_size / 1024 / 1024:.2f} MB")

    return output_file, creation_time, file_size


def plot_kernel_timeline(run_dir, min_size=None, max_size=None):
    """
    Create interactive kernel timeline plots.

    If min_size and max_size are provided, creates a single plot for that size range.
    Otherwise, creates one plot per segment from BIC segmentation.

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

    # Find ROCProfiler directory
    hostname = socket.gethostname()
    rocprof_dir = os.path.join(run_dir, 'rocp', hostname)

    if not os.path.exists(rocprof_dir):
        print(f"  Error: ROCProfiler directory not found: {rocprof_dir}")
        return []

    # Collect data for all ranks
    print("\n  Loading data for all ranks...")
    rank_data = {}

    for rank, pid in sorted(rank_pid_map.items()):
        # Load timestamp events
        timestamp_file = os.path.join(run_dir, f'rank_{rank}_timestamps.txt')
        if not os.path.exists(timestamp_file):
            print(f"    Warning: Timestamp file not found for rank {rank}")
            continue

        timestamp_events = parse_timestamp_file(timestamp_file)

        # Load kernel traces
        kernel_trace_file = os.path.join(rocprof_dir, f'{pid}_kernel_trace.csv')
        if not os.path.exists(kernel_trace_file):
            print(f"    Warning: Kernel trace not found for rank {rank}")
            continue

        kernels = parse_rocprof_kernel_trace(kernel_trace_file)

        # Filter to only NCCL kernels for cleaner visualization
        nccl_kernels = [k for k in kernels if 'nccl' in k['kernel_name'].lower()]

        rank_data[rank] = {
            'timestamp_events': timestamp_events,
            'kernels': kernels,
            'nccl_kernels': nccl_kernels
        }

    if not rank_data:
        print("  Error: No data loaded")
        return []

    print(f"  ✓ Loaded data for {len(rank_data)} rank(s)")

    # Create plots based on mode
    results = []
    
    if use_size_range:
        # Create single plot for specified size range
        result = plot_kernel_timeline_size_range(
            run_dir, rank_data, num_ranks, min_size, max_size
        )
        if result and result[0]:  # Check if output_file is not None
            results.append(result)
    else:
        # Create plots for each segment
        for segment in segments:
            seg_idx = segment['segment']
            result = plot_kernel_timeline_segment(
                run_dir, rank_data, num_ranks, segment, seg_idx
            )
            if result and result[0]:  # Check if output_file is not None
                results.append(result)

    return results


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

    results = plot_kernel_timeline(args.run_dir, args.min_size, args.max_size)

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

