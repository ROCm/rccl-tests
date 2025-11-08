#!/usr/bin/env python3
"""
Correlate benchmark timestamps with ROCProfiler kernel traces.

This script:
1. Parses benchmark timestamp files (rank_N_timestamps.txt)
2. Parses ROCProfiler kernel trace CSVs (pid_kernel_trace.csv)
3. Correlates kernels with benchmark runs using strict containment
4. Generates timing CSVs (all_rankN.csv) for analysis

The correlation uses STRICT CONTAINMENT - kernels must fall completely
within the Tstart/Tend window (no tolerance).

Usage:
    python correlate_rocprof_timings.py <run_directory>
"""

import os
import sys
import json
import csv
import socket
from pathlib import Path
from collections import defaultdict


def parse_timestamp_file(filepath):
    """
    Parse benchmark timestamp file.
    
    Format:
        Tstart 0: 123456789: all_reduce_size_1024_oop_type_float_op_sum_root_0
        Tend 0: 123456790
    
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


def extract_config_info(config_string):
    """
    Extract size and inplace info from config string.
    
    Example: all_reduce_size_1024_oop_type_float_op_sum_root_0
    Returns: (size_bytes, inplace)
    """
    parts = config_string.split('_')
    
    size_bytes = 0
    inplace = 0
    
    for i, part in enumerate(parts):
        if part == 'size' and i + 1 < len(parts):
            try:
                size_bytes = int(parts[i + 1])
            except ValueError:
                pass
        elif part == 'oop':
            inplace = 0
        elif part == 'inp':
            inplace = 1
    
    return size_bytes, inplace


def correlate_timestamps_with_kernels(timestamp_events, kernels):
    """
    Correlate kernels with benchmark runs using STRICT CONTAINMENT.
    
    Kernels must fall completely within Tstart/Tend windows.
    No tolerance window - timestamps are accurate enough.
    """
    runs = []
    
    # Pair up Tstart/Tend events
    i = 0
    while i < len(timestamp_events) - 1:
        if (timestamp_events[i]['event_type'] == 'start' and 
            timestamp_events[i + 1]['event_type'] == 'end'):
            
            start_event = timestamp_events[i]
            end_event = timestamp_events[i + 1]
            
            tstart = start_event['timestamp_ns']
            tend = end_event['timestamp_ns']
            config = start_event['config']
            
            # Find kernels that fall STRICTLY within this window
            matched_kernels = []
            for kernel in kernels:
                if (kernel['begin_ns'] >= tstart and 
                    kernel['end_ns'] <= tend):
                    matched_kernels.append(kernel)
            
            runs.append({
                'tstart': tstart,
                'tend': tend,
                'config': config,
                'kernels': matched_kernels
            })
            
            i += 2
        else:
            i += 1
    
    return runs


def generate_timing_csv(runs, output_file):
    """
    Generate timing CSV from correlated runs.
    
    Output format: size_bytes,inplace,iteration,time_seconds
    Each kernel gets its own row (individual kernel durations).
    """
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['size_bytes', 'inplace', 'iteration', 'time_seconds'])
        
        for run in runs:
            size_bytes, inplace = extract_config_info(run['config'])
            
            # Write each kernel as a separate row
            for iteration, kernel in enumerate(run['kernels']):
                time_seconds = kernel['duration_ns'] / 1e9
                writer.writerow([size_bytes, inplace, iteration, time_seconds])


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Correlate benchmark timestamps with ROCProfiler kernel traces')
    parser.add_argument('run_dir',
                        help='Run directory containing timestamp and kernel trace data')
    
    args = parser.parse_args()
    run_dir = args.run_dir
    
    if not os.path.isdir(run_dir):
        print(f"Error: Directory not found: {run_dir}")
        return 1
    
    print(f"Correlating ROCProfiler timings for: {run_dir}")
    
    # Load rank-to-PID mapping
    rank_pid_file = os.path.join(run_dir, 'rank_pid_mapping.json')
    if not os.path.exists(rank_pid_file):
        print(f"Error: No rank_pid_mapping.json found")
        return 1
    
    with open(rank_pid_file, 'r') as f:
        rank_pid_map = json.load(f)
    
    rank_pid_map = {int(k): int(v) for k, v in rank_pid_map.items()}
    print(f"  Loaded rank-to-PID mapping: {len(rank_pid_map)} ranks")
    
    # Find ROCProfiler directory
    hostname = socket.gethostname()
    rocprof_dir = os.path.join(run_dir, 'rocp', hostname)
    
    if not os.path.exists(rocprof_dir):
        print(f"Error: ROCProfiler directory not found: {rocprof_dir}")
        return 1
    
    # Process each rank
    total_runs = 0
    total_kernels = 0
    
    for rank, pid in sorted(rank_pid_map.items()):
        print(f"\n  Processing rank {rank} (PID {pid})...")
        
        # Load timestamp events
        timestamp_file = os.path.join(run_dir, f'rank_{rank}_timestamps.txt')
        if not os.path.exists(timestamp_file):
            print(f"    Warning: Timestamp file not found")
            continue
        
        timestamp_events = parse_timestamp_file(timestamp_file)
        print(f"    Loaded {len(timestamp_events)} timestamp events")
        
        # Load kernel traces
        kernel_trace_file = os.path.join(rocprof_dir, f'{pid}_kernel_trace.csv')
        if not os.path.exists(kernel_trace_file):
            print(f"    Warning: Kernel trace not found")
            continue
        
        kernels = parse_rocprof_kernel_trace(kernel_trace_file)
        print(f"    Loaded {len(kernels)} kernel dispatches")
        
        # Correlate
        runs = correlate_timestamps_with_kernels(timestamp_events, kernels)
        print(f"    Correlated {len(runs)} benchmark runs")
        
        # Count matched kernels
        matched_kernels = sum(len(run['kernels']) for run in runs)
        print(f"    Matched {matched_kernels} kernels to benchmark runs")
        
        if matched_kernels == 0:
            print(f"    WARNING: No kernels matched! Check timestamp alignment.")
            continue
        
        # Generate CSV
        output_csv = os.path.join(run_dir, f'all_rank{rank}.csv')
        generate_timing_csv(runs, output_csv)
        print(f"    Generated: {output_csv}")
        
        total_runs += len(runs)
        total_kernels += matched_kernels
    
    print(f"\n{'='*80}")
    print(f"CORRELATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total benchmark runs: {total_runs}")
    print(f"Total kernels matched: {total_kernels}")
    print(f"Output CSVs: all_rank*.csv")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

