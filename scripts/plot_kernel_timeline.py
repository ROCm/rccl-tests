#!/usr/bin/env python3
"""
Plot interactive kernel timeline showing kernel execution across MPI ranks.

Creates a timeline visualization with:
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


def plot_kernel_timeline(run_dir):
    """
    Create interactive kernel timeline plot.
    """
    print(f"Creating kernel timeline for: {run_dir}")
    
    # Load rank-to-PID mapping
    rank_pid_file = os.path.join(run_dir, 'rank_pid_mapping.json')
    if not os.path.exists(rank_pid_file):
        print(f"Error: No rank_pid_mapping.json found")
        return None
    
    with open(rank_pid_file, 'r') as f:
        rank_pid_map = json.load(f)
    
    rank_pid_map = {int(k): int(v) for k, v in rank_pid_map.items()}
    num_ranks = len(rank_pid_map)
    print(f"  Found {num_ranks} ranks")
    
    # Find ROCProfiler directory
    hostname = socket.gethostname()
    rocprof_dir = os.path.join(run_dir, 'rocp', hostname)
    
    if not os.path.exists(rocprof_dir):
        print(f"Error: ROCProfiler directory not found: {rocprof_dir}")
        return None
    
    # Collect data for all ranks
    rank_data = {}
    all_timestamps = []
    
    for rank, pid in sorted(rank_pid_map.items()):
        print(f"  Loading rank {rank} (PID {pid})...")
        
        # Load timestamp events
        timestamp_file = os.path.join(run_dir, f'rank_{rank}_timestamps.txt')
        if not os.path.exists(timestamp_file):
            print(f"    Warning: Timestamp file not found")
            continue
        
        timestamp_events = parse_timestamp_file(timestamp_file)
        
        # Load kernel traces
        kernel_trace_file = os.path.join(rocprof_dir, f'{pid}_kernel_trace.csv')
        if not os.path.exists(kernel_trace_file):
            print(f"    Warning: Kernel trace not found")
            continue
        
        kernels = parse_rocprof_kernel_trace(kernel_trace_file)
        
        # Filter to only NCCL kernels for cleaner visualization
        nccl_kernels = [k for k in kernels if 'nccl' in k['kernel_name'].lower()]
        
        rank_data[rank] = {
            'timestamp_events': timestamp_events,
            'kernels': kernels,
            'nccl_kernels': nccl_kernels
        }
        
        all_timestamps.extend([e['timestamp_ns'] for e in timestamp_events])
        
        print(f"    Loaded {len(timestamp_events)} timestamp events, {len(kernels)} total kernels, {len(nccl_kernels)} NCCL kernels")
    
    if not rank_data:
        print("Error: No data loaded")
        return None
    
    # Find time range and set zero point
    all_times = all_timestamps.copy()
    for rank_info in rank_data.values():
        for k in rank_info['kernels']:
            all_times.extend([k['begin_ns'], k['end_ns']])
    
    t_min = min(all_times)
    t_max = max(all_times)
    
    print(f"  Time range: {(t_max - t_min) / 1e9:.2f} seconds")
    print(f"  Zero point: {t_min} ns")
    
    # Create figure with subplots (one per rank)
    fig = make_subplots(
        rows=num_ranks,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=[f"Rank {r}" for r in sorted(rank_data.keys())]
    )
    
    # Color schemes
    colors = {
        'nccl_kernel': 'rgb(31, 119, 180)',      # Blue for NCCL kernels
        'other_kernel': 'rgb(200, 200, 200)',    # Gray for other kernels
        'tstart': 'rgba(0, 255, 0, 0.5)',        # Green for Tstart
        'tend': 'rgba(255, 0, 0, 0.5)',          # Red for Tend
    }
    
    # Plot each rank
    for row_idx, rank in enumerate(sorted(rank_data.keys()), start=1):
        data = rank_data[rank]
        
        # Plot kernels as horizontal bars
        # Use NCCL kernels for main visualization
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
    fig.update_layout(
        title={
            'text': f"RCCL Kernel Timeline - {os.path.basename(run_dir)}",
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
    output_file = os.path.join(run_dir, 'kernel_timeline.html')
    fig.write_html(output_file)
    print(f"\nSaved interactive plot to: {output_file}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Create interactive kernel timeline visualization')
    parser.add_argument('run_dir', help='Run directory containing timing data')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.run_dir):
        print(f"Error: Directory not found: {args.run_dir}")
        return 1
    
    output_file = plot_kernel_timeline(args.run_dir)
    
    if output_file:
        print(f"\nOpen in browser: file://{os.path.abspath(output_file)}")
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())

