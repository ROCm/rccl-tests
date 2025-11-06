#!/usr/bin/env python3
"""
Run RCCL benchmark timing sweeps with ROCProfiler integration.

This script:
1. Runs RCCL benchmarks with size sweeps
2. Collects ROCProfiler kernel traces
3. Captures benchmark timestamps
4. Correlates timing data
5. Saves results to hostname-specific data directory

Usage:
    python run_timing_sweep.py <benchmark_name> [options]
    
Example:
    python run_timing_sweep.py all_reduce --ranks 8 --min-size 8 --max-size 1G
"""

import os
import sys
import subprocess
import argparse
import socket
import shutil
import json
from datetime import datetime
from pathlib import Path


def get_data_directory():
    """Get hostname-specific data directory."""
    hostname = socket.gethostname()
    data_dir = f"/work/lmeadows/rccl/data/{hostname}"
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def create_output_directory(benchmark_name):
    """Create timestamped output directory."""
    data_dir = get_data_directory()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(data_dir, f"run_{benchmark_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def calculate_min_size_for_benchmark(benchmark_name, num_ranks, datatype='float'):
    """
    Calculate minimum size to avoid zero-size outputs due to alignment.
    
    Affected benchmarks apply alignment mask: & -(16/eltSize)
    For float (4 bytes): mask is & -4, which zeros lower 2 bits
    """
    # Element size for float
    elt_size = 4
    
    # Benchmarks affected by alignment issue
    affected_benchmarks = [
        'alltoall',
        'all_gather',
        'gather',
        'reduce_scatter',
        'scatter',
        'sendrecv'
    ]
    
    if benchmark_name not in affected_benchmarks:
        return 8  # Default minimum
    
    # For affected benchmarks, calculate safe minimum
    # The formula is: size_per_rank = (total_size / num_ranks) & -(16/eltSize)
    # We need: (min_size / num_ranks) & -4 >= 8
    # So: min_size >= num_ranks * 8 (rounded up to next multiple of 4*num_ranks)
    
    min_size = num_ranks * 8
    alignment = 16 // elt_size  # 4 for float
    min_size = ((min_size + alignment - 1) // alignment) * alignment
    
    # Make sure it's at least 16 bytes
    min_size = max(min_size, 16)
    
    return min_size


def parse_size(size_str):
    """Parse size string like '8', '1K', '1M', '1G' to bytes."""
    size_str = size_str.upper().strip()
    
    multipliers = {
        'K': 1024,
        'M': 1024 * 1024,
        'G': 1024 * 1024 * 1024,
    }
    
    if size_str[-1] in multipliers:
        return int(size_str[:-1]) * multipliers[size_str[-1]]
    else:
        return int(size_str)


def generate_size_sweep(min_size, max_size):
    """Generate size sweep from min to max, doubling each step."""
    sizes = []
    current = min_size
    while current <= max_size:
        sizes.append(current)
        current *= 2
    return sizes


def cleanup_old_timing_files(output_dir):
    """Remove any existing CSV files before a run."""
    for f in Path(output_dir).glob("*.csv"):
        f.unlink()


def parse_rank_pid_mapping(stdout_output):
    """
    Extract rank-to-PID mapping from benchmark output.
    
    Looks for lines like:
    # Rank 0 Group 0 Pid 1234567 on hostname device 0 [0000:00:00] AMD Instinct MI350X
    """
    rank_pid_map = {}
    
    for line in stdout_output.split('\n'):
        if '# Rank' in line and 'Pid' in line:
            parts = line.split()
            try:
                rank_idx = parts.index('Rank') + 1
                pid_idx = parts.index('Pid') + 1
                rank = int(parts[rank_idx])
                pid = int(parts[pid_idx])
                rank_pid_map[rank] = pid
            except (ValueError, IndexError):
                continue
    
    return rank_pid_map


def run_benchmark_for_full_range(benchmark_name, num_ranks, min_size, max_size, 
                                  output_dir, iterations=100, warmup=5):
    """
    Run benchmark with ROCProfiler for full size range.
    """
    # Benchmark binary path
    benchmark_path = f"/work/lmeadows/rccl/rccl-tests/build/{benchmark_name}_perf"
    
    if not os.path.exists(benchmark_path):
        print(f"Error: Benchmark not found: {benchmark_path}")
        return None
    
    # Create rocprof output directory
    hostname = socket.gethostname()
    rocprof_dir = os.path.join(output_dir, 'rocp')
    os.makedirs(rocprof_dir, exist_ok=True)
    
    # Build command
    cmd = [
        'mpirun',
        '-np', str(num_ranks),
        '--bind-to', 'numa',
        '/opt/rocm/bin/rocprofv3',
        '--kernel-trace',
        '-f', 'csv',
        '-d', rocprof_dir,
        '--',
        benchmark_path,
        '-b', str(min_size),
        '-e', str(max_size),
        '-f', '2',
        '-n', str(iterations),
        '-w', str(warmup),
        '-g', '1'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Output directory: {output_dir}")
    print(f"ROCProfiler output: {rocprof_dir}")
    
    # Run benchmark
    try:
        result = subprocess.run(
            cmd,
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        # Save stdout and stderr
        with open(os.path.join(output_dir, f'{benchmark_name}_benchmark_output.txt'), 'w') as f:
            f.write(result.stdout)
        
        with open(os.path.join(output_dir, f'{benchmark_name}_benchmark_stderr.txt'), 'w') as f:
            f.write(result.stderr)
        
        if result.returncode != 0:
            print(f"Warning: Benchmark exited with code {result.returncode}")
            print(f"Stderr: {result.stderr[:500]}")
        
        return result.stdout
        
    except subprocess.TimeoutExpired:
        print("Error: Benchmark timed out after 1 hour")
        return None
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return None


def collect_timing_files(output_dir, num_ranks):
    """
    Collect timing files from benchmark run.
    
    Expected files:
    - rank_N_timestamps.txt (from benchmark)
    - rocp/<hostname>/<pid>_kernel_trace.csv (from rocprofv3)
    """
    hostname = socket.gethostname()
    rocprof_dir = os.path.join(output_dir, 'rocp', hostname)
    
    # Check for timestamp files
    timestamp_files = []
    for rank in range(num_ranks):
        ts_file = os.path.join(output_dir, f'rank_{rank}_timestamps.txt')
        if os.path.exists(ts_file):
            timestamp_files.append(ts_file)
    
    # Check for rocprof files
    rocprof_files = []
    if os.path.exists(rocprof_dir):
        rocprof_files = list(Path(rocprof_dir).glob('*_kernel_trace.csv'))
    
    print(f"\nCollected files:")
    print(f"  Timestamp files: {len(timestamp_files)}/{num_ranks}")
    print(f"  ROCProf traces: {len(rocprof_files)}")
    
    return len(timestamp_files) == num_ranks and len(rocprof_files) > 0


def main():
    parser = argparse.ArgumentParser(
        description='Run RCCL benchmark timing sweep with ROCProfiler')
    parser.add_argument('benchmark', help='Benchmark name (e.g., all_reduce)')
    parser.add_argument('--ranks', type=int, default=8,
                       help='Number of MPI ranks (default: 8)')
    parser.add_argument('--min-size', type=str, default='8',
                       help='Minimum message size (default: 8)')
    parser.add_argument('--max-size', type=str, default='1G',
                       help='Maximum message size (default: 1G)')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of timed iterations (default: 100)')
    parser.add_argument('--warmup', type=int, default=5,
                       help='Number of warmup iterations (default: 5)')
    
    args = parser.parse_args()
    
    # Parse sizes
    min_size = parse_size(args.min_size)
    max_size = parse_size(args.max_size)
    
    # Calculate adjusted minimum size for affected benchmarks
    calculated_min = calculate_min_size_for_benchmark(args.benchmark, args.ranks)
    if calculated_min > min_size:
        print(f"Note: Adjusting minimum size from {min_size} to {calculated_min} bytes")
        print(f"      to avoid zero-size outputs for {args.benchmark} with {args.ranks} ranks")
        min_size = calculated_min
    
    # Create output directory
    output_dir = create_output_directory(args.benchmark)
    print(f"\n{'='*80}")
    print(f"RCCL Benchmark Timing Sweep")
    print(f"{'='*80}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Ranks: {args.ranks}")
    print(f"Size range: {min_size} - {max_size} bytes")
    print(f"Iterations: {args.iterations} (warmup: {args.warmup})")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")
    
    # Clean up old timing files
    cleanup_old_timing_files(output_dir)
    
    # Run benchmark
    stdout = run_benchmark_for_full_range(
        args.benchmark,
        args.ranks,
        min_size,
        max_size,
        output_dir,
        args.iterations,
        args.warmup
    )
    
    if stdout is None:
        print("\nError: Benchmark run failed")
        return 1
    
    # Parse rank-to-PID mapping
    rank_pid_map = parse_rank_pid_mapping(stdout)
    if not rank_pid_map:
        print("Warning: Could not parse rank-to-PID mapping")
    else:
        # Save mapping to JSON
        mapping_file = os.path.join(output_dir, 'rank_pid_mapping.json')
        with open(mapping_file, 'w') as f:
            json.dump(rank_pid_map, f, indent=2)
        print(f"\nSaved rank-to-PID mapping: {len(rank_pid_map)} ranks")
    
    # Collect timing files
    success = collect_timing_files(output_dir, args.ranks)
    
    if not success:
        print("\nWarning: Not all expected timing files were collected")
        return 1
    
    # Run correlation script
    print("\nCorrelating ROCProfiler timings...")
    correlate_script = os.path.join(
        os.path.dirname(__file__),
        'correlate_rocprof_timings.py'
    )
    
    if os.path.exists(correlate_script):
        try:
            result = subprocess.run(
                ['python3', correlate_script, output_dir],
                capture_output=True,
                text=True,
                timeout=600
            )
            
            print(result.stdout)
            
            if result.returncode != 0:
                print(f"Warning: Correlation script failed")
                print(result.stderr)
                return 1
                
        except Exception as e:
            print(f"Error running correlation script: {e}")
            return 1
    else:
        print(f"Warning: Correlation script not found: {correlate_script}")
        return 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"Benchmark output: {args.benchmark}_benchmark_output.txt")
    print(f"Timing CSVs: all_rank*.csv")
    print(f"ROCProf traces: rocp/{socket.gethostname()}/*_kernel_trace.csv")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

