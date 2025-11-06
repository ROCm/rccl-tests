#!/usr/bin/env python3
"""
Analyze NCCL kernel counts and durations between benchmark markers.

This script:
1. Loads benchmark timestamps (Tstart/Tend)
2. Loads ROCProfiler kernel traces
3. Correlates kernels with benchmark runs
4. Generates tables showing kernel counts and durations per size/rank

Usage:
    python analyze_nccl_kernel_counts.py <run_directory>
"""

import os
import sys
import json
import csv
import socket
import argparse
from collections import defaultdict
import pandas as pd


def parse_timestamp_file(filepath):
    """Parse benchmark timestamp file."""
    events = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('Tstart'):
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
    """Parse ROCProfiler kernel trace CSV."""
    kernels = []
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
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
            except (ValueError, KeyError):
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


def correlate_kernels(timestamp_events, kernels):
    """Correlate kernels with benchmark runs using strict containment."""
    runs = []
    
    i = 0
    while i < len(timestamp_events) - 1:
        if (timestamp_events[i]['event_type'] == 'start' and 
            timestamp_events[i + 1]['event_type'] == 'end'):
            
            start_event = timestamp_events[i]
            end_event = timestamp_events[i + 1]
            
            tstart = start_event['timestamp_ns']
            tend = end_event['timestamp_ns']
            config = start_event['config']
            
            # Find kernels strictly within this window
            matched_kernels = []
            for kernel in kernels:
                if (kernel['begin_ns'] >= tstart and 
                    kernel['end_ns'] <= tend):
                    matched_kernels.append(kernel)
            
            runs.append({
                'tstart': tstart,
                'tend': tend,
                'config': config,
                'size_bytes': extract_size_from_config(config),
                'kernels': matched_kernels
            })
            
            i += 2
        else:
            i += 1
    
    return runs


def analyze_kernel_counts(run_dir):
    """Analyze NCCL kernel counts across all ranks."""
    
    print(f"Analyzing NCCL kernel counts for: {run_dir}")
    
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
    all_runs = defaultdict(lambda: defaultdict(list))
    
    for rank, pid in sorted(rank_pid_map.items()):
        print(f"  Processing rank {rank} (PID {pid})...")
        
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
        
        # Filter to NCCL kernels
        nccl_kernels = [k for k in kernels if 'ncclDevKernel_Generic' in k['kernel_name']]
        
        # Correlate
        runs = correlate_kernels(timestamp_events, nccl_kernels)
        
        # Organize by size
        for run in runs:
            size = run['size_bytes']
            kernel_count = len(run['kernels'])
            durations = [k['duration_ns'] / 1000.0 for k in run['kernels']]  # Convert to microseconds
            
            all_runs[size][rank].append({
                'count': kernel_count,
                'durations': durations,
                'min_us': min(durations) if durations else 0,
                'max_us': max(durations) if durations else 0,
                'avg_us': sum(durations) / len(durations) if durations else 0
            })
        
        print(f"    Processed {len(runs)} benchmark runs")
    
    return all_runs


def print_summary_table(all_runs):
    """Print summary table of kernel counts and durations."""
    
    print("\n" + "="*100)
    print("NCCL KERNEL COUNT AND DURATION ANALYSIS")
    print("="*100)
    
    # Get all sizes sorted
    sizes = sorted(all_runs.keys())
    
    for size in sizes:
        print(f"\nSize: {size} bytes")
        print("-" * 100)
        
        # Get all ranks for this size
        ranks = sorted(all_runs[size].keys())
        
        # Create table
        rows = []
        for rank in ranks:
            runs = all_runs[size][rank]
            
            # Average across all runs for this rank/size
            avg_count = sum(r['count'] for r in runs) / len(runs)
            all_durations = []
            for r in runs:
                all_durations.extend(r['durations'])
            
            if all_durations:
                min_dur = min(all_durations)
                max_dur = max(all_durations)
                avg_dur = sum(all_durations) / len(all_durations)
            else:
                min_dur = max_dur = avg_dur = 0
            
            rows.append({
                'Rank': rank,
                'Avg Count': f"{avg_count:.1f}",
                'Min Duration (µs)': f"{min_dur:.2f}",
                'Max Duration (µs)': f"{max_dur:.2f}",
                'Avg Duration (µs)': f"{avg_dur:.2f}"
            })
        
        # Print as DataFrame
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
    
    print("\n" + "="*100)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze NCCL kernel counts and durations')
    parser.add_argument('run_dir', help='Run directory containing timing data')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.run_dir):
        print(f"Error: Directory not found: {args.run_dir}")
        return 1
    
    all_runs = analyze_kernel_counts(args.run_dir)
    
    if all_runs:
        print_summary_table(all_runs)
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())

