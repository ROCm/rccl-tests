#!/usr/bin/env python3
"""
Verify the relationship between wall-clock time and timestamp deltas.

Expected relationship:
  wall_clock_time = time to launch kernels (CPU-side)
  timestamp_delta = wall_clock_time + GPU synchronization overhead
  
Therefore:
  timestamp_delta/iterations > wall_clock_time (reported in benchmark output)
  difference = GPU sync overhead

Usage:
    python verify_timing_relationship.py <run_directory>
"""

import os
import sys
import argparse
import re
import pandas as pd
import numpy as np


def parse_benchmark_output(filepath):
    """
    Parse benchmark output to extract wall-clock times.
    
    Returns: DataFrame with columns: size_bytes, operation_mode, wall_time_us
    """
    records = []
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find the data table
    lines = content.strip().split('\n')
    
    in_data_section = False
    for line in lines:
        line = line.strip()
        
        # Skip header lines
        if line.startswith('#'):
            if 'size' in line and 'time' in line:
                in_data_section = True
            continue
        
        if not in_data_section:
            continue
        
        if not line:
            continue
        
        # Parse data line
        parts = line.split()
        if len(parts) < 6:
            continue
        
        try:
            size_bytes = int(parts[0])
            time_us = float(parts[5])
            
            # Determine operation mode from previous context or assume out-of-place first
            # For now, we'll extract this from timestamp correlation
            records.append({
                'size_bytes': size_bytes,
                'wall_time_us': time_us
            })
        except (ValueError, IndexError):
            continue
    
    return pd.DataFrame(records)


def parse_timestamp_file(filepath):
    """
    Parse timestamp file and pair Tstart/Tend events.
    
    Returns: DataFrame with columns: size_bytes, operation_mode, start_ns, end_ns, delta_ns
    """
    events = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('Tstart'):
                # Tstart 0: 123456789: AllReduce_size_1024_oop_type_float_op_sum_root_-1
                parts = line.split(':', 2)
                if len(parts) >= 3:
                    rank = int(parts[0].replace('Tstart', '').strip())
                    timestamp = int(parts[1].strip())
                    config = parts[2].strip()
                    
                    # Extract size
                    size_match = re.search(r'size_(\d+)', config)
                    size_bytes = int(size_match.group(1)) if size_match else 0
                    
                    # Extract operation mode
                    operation_mode = 'inp' if '_inp_' in config else 'oop'
                    
                    events.append({
                        'rank': rank,
                        'timestamp_ns': timestamp,
                        'event_type': 'start',
                        'size_bytes': size_bytes,
                        'operation_mode': operation_mode
                    })
            
            elif line.startswith('Tend'):
                # Tend 0: 123456789
                parts = line.split(':')
                if len(parts) >= 2:
                    rank = int(parts[0].replace('Tend', '').strip())
                    timestamp = int(parts[1].strip())
                    
                    events.append({
                        'rank': rank,
                        'timestamp_ns': timestamp,
                        'event_type': 'end'
                    })
    
    # Pair up Tstart/Tend events
    pairs = []
    i = 0
    while i < len(events) - 1:
        if events[i]['event_type'] == 'start' and events[i+1]['event_type'] == 'end':
            start_event = events[i]
            end_event = events[i+1]
            
            if start_event['rank'] == end_event['rank']:
                pairs.append({
                    'rank': start_event['rank'],
                    'size_bytes': start_event['size_bytes'],
                    'operation_mode': start_event['operation_mode'],
                    'start_ns': start_event['timestamp_ns'],
                    'end_ns': end_event['timestamp_ns'],
                    'delta_ns': end_event['timestamp_ns'] - start_event['timestamp_ns']
                })
            i += 2
        else:
            i += 1
    
    return pd.DataFrame(pairs)


def analyze_timing_relationship(run_dir):
    """
    Analyze the relationship between wall-clock time and timestamp deltas.
    """
    print(f"Analyzing timing relationship for: {run_dir}")
    print("=" * 80)
    
    # Find benchmark output file
    benchmark_files = [f for f in os.listdir(run_dir) if f.endswith('_benchmark_output.txt')]
    if not benchmark_files:
        print("Error: No benchmark output file found")
        return 1
    
    benchmark_file = os.path.join(run_dir, benchmark_files[0])
    benchmark_name = benchmark_files[0].replace('_benchmark_output.txt', '')
    
    print(f"Benchmark: {benchmark_name}")
    print(f"Run directory: {run_dir}\n")
    
    # Load benchmark wall-clock times
    df_wall = parse_benchmark_output(benchmark_file)
    if df_wall.empty:
        print("Error: Could not parse benchmark output")
        return 1
    
    print(f"Loaded {len(df_wall)} wall-clock measurements from benchmark output\n")
    
    # Load timestamp data from all ranks
    timestamp_files = [f for f in os.listdir(run_dir) if f.startswith('rank_') and f.endswith('_timestamps.txt')]
    if not timestamp_files:
        print("Error: No timestamp files found")
        return 1
    
    print(f"Found {len(timestamp_files)} timestamp files\n")
    
    # Load and combine timestamp data
    df_timestamps_list = []
    for ts_file in sorted(timestamp_files):
        ts_path = os.path.join(run_dir, ts_file)
        df_ts = parse_timestamp_file(ts_path)
        if not df_ts.empty:
            df_timestamps_list.append(df_ts)
    
    if not df_timestamps_list:
        print("Error: Could not parse timestamp files")
        return 1
    
    df_timestamps = pd.concat(df_timestamps_list, ignore_index=True)
    print(f"Loaded {len(df_timestamps)} timestamp pairs\n")
    
    # Get metadata to find iterations
    metadata_file = os.path.join(run_dir, 'run_metadata.json')
    iterations = 100  # default
    if os.path.exists(metadata_file):
        import json
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            iterations = metadata.get('iterations', 100)
    
    print(f"Iterations per size: {iterations}\n")
    
    # Compute average timestamp delta per (size, operation_mode)
    df_ts_avg = df_timestamps.groupby(['size_bytes', 'operation_mode']).agg({
        'delta_ns': ['mean', 'std', 'min', 'max', 'count']
    }).reset_index()
    
    df_ts_avg.columns = ['size_bytes', 'operation_mode', 'delta_ns_mean', 'delta_ns_std', 
                          'delta_ns_min', 'delta_ns_max', 'rank_count']
    
    # Convert delta to per-iteration time in microseconds
    df_ts_avg['timestamp_time_us'] = df_ts_avg['delta_ns_mean'] / 1000.0 / iterations
    df_ts_avg['timestamp_time_std_us'] = df_ts_avg['delta_ns_std'] / 1000.0 / iterations
    
    # Merge with wall-clock times
    # Note: benchmark output doesn't distinguish inp/oop, so we compare against both
    comparison_records = []
    
    for _, ts_row in df_ts_avg.iterrows():
        size = ts_row['size_bytes']
        mode = ts_row['operation_mode']
        ts_time_us = ts_row['timestamp_time_us']
        ts_std_us = ts_row['timestamp_time_std_us']
        
        # Find matching wall-clock time
        wall_matches = df_wall[df_wall['size_bytes'] == size]
        if not wall_matches.empty:
            wall_time_us = wall_matches.iloc[0]['wall_time_us']
            
            sync_overhead_us = ts_time_us - wall_time_us
            overhead_percent = (sync_overhead_us / wall_time_us * 100) if wall_time_us > 0 else 0
            
            comparison_records.append({
                'size_bytes': size,
                'operation_mode': mode,
                'wall_time_us': wall_time_us,
                'timestamp_time_us': ts_time_us,
                'timestamp_std_us': ts_std_us,
                'sync_overhead_us': sync_overhead_us,
                'overhead_percent': overhead_percent,
                'ranks': int(ts_row['rank_count'])
            })
    
    df_comparison = pd.DataFrame(comparison_records)
    
    if df_comparison.empty:
        print("Error: Could not match wall-clock and timestamp data")
        return 1
    
    # Print results
    print("=" * 80)
    print("TIMING RELATIONSHIP ANALYSIS")
    print("=" * 80)
    print()
    print("Expected: timestamp_time = wall_time + GPU_sync_overhead")
    print("Therefore: timestamp_time > wall_time")
    print()
    print("=" * 80)
    
    # Format output
    def format_size(size_bytes):
        if size_bytes >= 1024*1024*1024:
            return f"{size_bytes / (1024*1024*1024):.1f} GiB"
        elif size_bytes >= 1024*1024:
            return f"{size_bytes / (1024*1024):.1f} MiB"
        elif size_bytes >= 1024:
            return f"{size_bytes / 1024:.1f} KiB"
        else:
            return f"{size_bytes} B"
    
    # Print table
    print(f"{'Size':<12} {'Mode':<5} {'Wall(μs)':<12} {'Tstamp(μs)':<12} {'Sync(μs)':<12} {'Ovhd%':<8} {'Ranks':<6}")
    print("-" * 80)
    
    for _, row in df_comparison.iterrows():
        size_str = format_size(row['size_bytes'])
        print(f"{size_str:<12} {row['operation_mode']:<5} "
              f"{row['wall_time_us']:>10.2f}  "
              f"{row['timestamp_time_us']:>10.2f}  "
              f"{row['sync_overhead_us']:>10.2f}  "
              f"{row['overhead_percent']:>6.1f}%  "
              f"{row['ranks']:>4}")
    
    print()
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # Overall statistics
    print(f"\nMean sync overhead: {df_comparison['sync_overhead_us'].mean():.2f} μs")
    print(f"Median sync overhead: {df_comparison['sync_overhead_us'].median():.2f} μs")
    print(f"Std dev sync overhead: {df_comparison['sync_overhead_us'].std():.2f} μs")
    print(f"Min sync overhead: {df_comparison['sync_overhead_us'].min():.2f} μs")
    print(f"Max sync overhead: {df_comparison['sync_overhead_us'].max():.2f} μs")
    
    print(f"\nMean overhead percentage: {df_comparison['overhead_percent'].mean():.2f}%")
    print(f"Median overhead percentage: {df_comparison['overhead_percent'].median():.2f}%")
    
    # Check for anomalies (negative overhead)
    negative_overhead = df_comparison[df_comparison['sync_overhead_us'] < 0]
    if not negative_overhead.empty:
        print(f"\n⚠️  WARNING: Found {len(negative_overhead)} cases with NEGATIVE sync overhead!")
        print("This suggests timestamp_time < wall_time, which violates expectations.")
        print("\nNegative overhead cases:")
        for _, row in negative_overhead.iterrows():
            print(f"  Size {format_size(row['size_bytes'])}, {row['operation_mode']}: "
                  f"{row['sync_overhead_us']:.2f} μs ({row['overhead_percent']:.1f}%)")
    else:
        print("\n✓ All sync overheads are positive (as expected)")
    
    # Group by operation mode
    print("\n" + "=" * 80)
    print("BY OPERATION MODE")
    print("=" * 80)
    
    for mode in ['oop', 'inp']:
        mode_data = df_comparison[df_comparison['operation_mode'] == mode]
        if not mode_data.empty:
            mode_label = "Out-of-Place" if mode == 'oop' else "In-Place"
            print(f"\n{mode_label}:")
            print(f"  Mean sync overhead: {mode_data['sync_overhead_us'].mean():.2f} μs")
            print(f"  Mean overhead percentage: {mode_data['overhead_percent'].mean():.2f}%")
            print(f"  Samples: {len(mode_data)}")
    
    print("\n" + "=" * 80)
    
    # Save detailed results
    output_file = os.path.join(run_dir, 'timing_relationship_analysis.csv')
    df_comparison.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Verify timing relationship between wall-clock and timestamp deltas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Expected relationship:
  wall_clock_time    = kernel launch time (CPU-side only)
  timestamp_delta    = wall_clock_time + GPU sync overhead
  sync_overhead      = timestamp_delta - wall_clock_time

Therefore: timestamp_delta should be > wall_clock_time

This script calculates and reports the GPU synchronization overhead.
        """
    )
    parser.add_argument('run_dir', help='Run directory containing timing data')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.run_dir):
        print(f"Error: Directory not found: {args.run_dir}")
        return 1
    
    return analyze_timing_relationship(args.run_dir)


if __name__ == '__main__':
    sys.exit(main())



