#!/usr/bin/env python3
"""
Analyze RCCL timing sweep data and generate statistical summaries.
Creates comprehensive statistical analysis of individual kernel timings.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import re

def load_timing_data(output_dir):
    """Load all timing CSV files from a benchmark run directory

    Returns:
        pd.DataFrame: Combined timing data from all ranks
    """
    timing_files = glob.glob(os.path.join(output_dir, "*_rank*.csv"))

    if not timing_files:
        print(f"No timing files found in {output_dir}")
        return None

    print(f"Loading {len(timing_files)} timing files...")

    # Load and concatenate all rank data
    dfs = []
    for file in timing_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {file}: {e}")
            continue

    if not dfs:
        print("No valid timing files could be loaded")
        return None

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined data: {len(combined_df)} rows from {len(dfs)} rank files")
    return combined_df

def load_benchmark_output(output_dir, benchmark_name):
    """Load benchmark output file content

    Returns:
        str: Raw benchmark output content, or None if file not found
    """
    output_file = os.path.join(output_dir, f"{benchmark_name}_benchmark_output.txt")

    if not os.path.exists(output_file):
        print(f"Benchmark output file not found: {output_file}")
        return None

    # Read the raw output
    with open(output_file, 'r') as f:
        content = f.read()

    print(f"Loaded benchmark output ({len(content)} chars) from {output_file}")
    return content

def parse_benchmark_output(content):
    """Parse benchmark output to extract wall clock timing information per size

    Args:
        content (str): Raw benchmark output

    Returns:
        pd.DataFrame: DataFrame with size_bytes and wall_time_us columns
    """
    lines = content.strip().split('\n')
    results = []

    # Pattern to match benchmark timing lines
    # Looking for lines like: "1024           256     float     sum      -1    37.15    0.02    0.00      0"
    timing_pattern = re.compile(r'^\s*(\d+)\s+(\d+)\s+(\w+)\s+(\w+)\s+(-?\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)')

    for line in lines:
        match = timing_pattern.match(line)
        if match:
            try:
                size_bytes = int(match.group(1))
                count = int(match.group(2))
                data_type = match.group(3)
                op = match.group(4)
                root = int(match.group(5))
                time_us = float(match.group(6))
                alg_bw = float(match.group(7))
                bus_bw = float(match.group(8))
                errors = int(match.group(9))

                results.append({
                    'size_bytes': size_bytes,
                    'count': count,
                    'data_type': data_type,
                    'operation': op,
                    'root': root,
                    'wall_time_us': time_us,
                    'alg_bw_gbs': alg_bw,
                    'bus_bw_gbs': bus_bw,
                    'errors': errors
                })
            except (ValueError, IndexError) as e:
                print(f"Warning: Failed to parse timing line: {line.strip()}")
                continue

    if results:
        df = pd.DataFrame(results)
        print(f"Parsed {len(df)} timing entries from benchmark output")
        return df
    else:
        print("No timing data found in benchmark output")
        return pd.DataFrame()

def create_statistical_summary(timing_df, benchmark_df=None):
    """Create statistical summary of individual kernel timing data

    Args:
        timing_df (pd.DataFrame): Individual kernel timings from CSV files
        benchmark_df (pd.DataFrame): Wall clock timings from benchmark output

    Returns:
        pd.DataFrame: Statistical summary with one row per size/operation combination
    """

    summary_stats = []

    # Group by size and inplace flag
    for inplace_val in sorted(timing_df['inplace'].unique()):
        inplace_df = timing_df[timing_df['inplace'] == inplace_val]
        inplace_label = "in-place" if inplace_val else "out-of-place"

        for size in sorted(inplace_df['size_bytes'].unique()):
            size_df = inplace_df[inplace_df['size_bytes'] == size]

            if len(size_df) == 0:
                continue

            # Calculate statistics on timing data (convert to microseconds)
            times_us = size_df['time_seconds'] * 1e6

            stats = {
                'size_bytes': size,
                'operation': inplace_label,
                'kernel_count': len(times_us),
                'kernel_mean_us': times_us.mean(),
                'kernel_std_us': times_us.std(),
                'kernel_min_us': times_us.min(),
                'kernel_max_us': times_us.max(),
                'kernel_p25_us': times_us.quantile(0.25),
                'kernel_p50_us': times_us.quantile(0.50),
                'kernel_p75_us': times_us.quantile(0.75),
                'kernel_p95_us': times_us.quantile(0.95),
                'kernel_p99_us': times_us.quantile(0.99),
                'kernel_cv_percent': (times_us.std() / times_us.mean() * 100) if times_us.mean() > 0 else 0
            }

            # Add benchmark wall clock time if available
            if benchmark_df is not None and not benchmark_df.empty:
                # Match by size (benchmark output may have different granularity)
                wall_matches = benchmark_df[benchmark_df['size_bytes'] == size]
                if len(wall_matches) > 0:
                    stats['wall_time_us'] = wall_matches['wall_time_us'].mean()
                    stats['wall_errors'] = wall_matches['errors'].mean()
                else:
                    # Try to find closest size match for interpolation
                    available_sizes = sorted(benchmark_df['size_bytes'].unique())
                    closest_size = min(available_sizes, key=lambda x: abs(x - size))
                    if abs(closest_size - size) / size < 0.1:  # Within 10%
                        wall_matches = benchmark_df[benchmark_df['size_bytes'] == closest_size]
                        if len(wall_matches) > 0:
                            stats['wall_time_us'] = wall_matches['wall_time_us'].mean()
                            stats['wall_errors'] = wall_matches['errors'].mean()
                            stats['wall_size_approx'] = True

            summary_stats.append(stats)

    result_df = pd.DataFrame(summary_stats)

    # Sort by size, then by operation
    result_df = result_df.sort_values(['size_bytes', 'operation']).reset_index(drop=True)

    return result_df

def format_size(size_bytes):
    """Format size in human readable format"""
    if size_bytes >= 1024**3:
        return "6.1f"
    elif size_bytes >= 1024**2:
        return "6.1f"
    elif size_bytes >= 1024:
        return "6.1f"
    else:
        return "6.0f"

def print_summary_table(summary_df, benchmark_name):
    """Print formatted statistical summary table"""

    print(f"\n{'='*100}")
    print(f"RCCL {benchmark_name.upper()} - Statistical Timing Summary")
    print(f"{'='*100}")
    print(f"Individual kernel timings aggregated across all ranks")
    print(f"{'='*100}")

    # Group by operation type
    for operation in summary_df['operation'].unique():
        op_df = summary_df[summary_df['operation'] == operation]

        print(f"\n{operation.upper()} Operations:")
        print("-" * 80)

        # Print header
        header = "7"
        print(header)
        print("-" * len(header))

        for _, row in op_df.iterrows():
            size_str = format_size(row['size_bytes'])
            count_str = "4d"

            # Wall time column
            if pd.isna(row.get('wall_time_us', pd.NA)):
                wall_str = "N/A    "
            else:
                wall_approx = "*" if row.get('wall_size_approx', False) else " "
                wall_str = "6.1f"

            # Kernel timing columns
            line = "6.1f"
            print(line)

    print(f"\n{'='*100}")
    print("Notes:")
    print("- Kernel timings: Individual GPU kernel execution times")
    print("- Wall timings: End-to-end application wall clock times")
    print("- CV: Coefficient of variation (std/mean * 100%)")
    print("- * : Approximated wall time (closest size match)")
    print("- All times in microseconds (μs)")

def save_detailed_results(summary_df, output_dir, benchmark_name):
    """Save detailed results to CSV file"""
    output_file = os.path.join(output_dir, f"{benchmark_name}_timing_analysis.csv")
    summary_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    return output_file

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze RCCL timing sweep data and generate statistical summaries')
    parser.add_argument('run_dir', 
                        help='Run directory containing timing data (e.g., run_all_gather_20251102_085929)')
    
    args = parser.parse_args()
    output_dir = args.run_dir

    if not os.path.exists(output_dir):
        print(f"Directory not found: {output_dir}")
        sys.exit(1)

    # Extract benchmark name from directory
    dir_name = os.path.basename(output_dir)
    if not dir_name.startswith('run_'):
        print("Directory name should start with 'run_'")
        print(f"Got: {dir_name}")
        sys.exit(1)

    # Extract benchmark name (everything between 'run_' and the timestamp)
    # Format: run_{benchmark_name}_{YYYYMMDD_HHMMSS}
    # Timestamp is 15 characters: 8 digits + underscore + 6 digits
    import re
    timestamp_pattern = r'_\d{8}_\d{6}$'
    match = re.search(timestamp_pattern, dir_name)

    if not match:
        print(f"Cannot find timestamp pattern in directory name: {dir_name}")
        print("Expected format: run_{benchmark_name}_{YYYYMMDD_HHMMSS}")
        sys.exit(1)

    timestamp_start = match.start()
    benchmark_name = dir_name[4:timestamp_start]  # Skip 'run_' prefix

    print(f"Analyzing {benchmark_name} benchmark data from {output_dir}")

    # Load timing data
    timing_df = load_timing_data(output_dir)
    if timing_df is None:
        sys.exit(1)

    # Load benchmark output
    benchmark_content = load_benchmark_output(output_dir, benchmark_name)
    benchmark_df = None
    if benchmark_content:
        benchmark_df = parse_benchmark_output(benchmark_content)

    # Create statistical summary
    summary_df = create_statistical_summary(timing_df, benchmark_df)

    if summary_df.empty:
        print("No statistical summary could be generated")
        sys.exit(1)

    # Print results
    print_summary_table(summary_df, benchmark_name)

    # Save to CSV
    csv_file = save_detailed_results(summary_df, output_dir, benchmark_name)

    print(f"\nAnalysis complete. Summary table displayed above, detailed data in {csv_file}")

if __name__ == "__main__":
    main()
