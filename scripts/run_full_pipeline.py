#!/usr/bin/env python3
"""
Run complete RCCL performance analysis pipeline.

This script orchestrates the entire analysis workflow:
1. Run benchmark with timing sweep
2. Correlate ROCProfiler kernel traces
3. Generate statistical analysis
4. Perform BIC segmentation
5. Create interactive visualizations
6. Generate boxplots

Note: plot_kernel_timeline.py is not included (needs optimization).
      Run manually if needed: python3 plot_kernel_timeline.py <run_dir>

Usage:
    python3 run_full_pipeline.py <benchmark> [options]

Examples:
    # Basic run with defaults (8 ranks)
    python3 run_full_pipeline.py all_reduce

    # Custom configuration
    python3 run_full_pipeline.py all_reduce --ranks 4 --iterations 50

    # Skip visualizations (faster)
    python3 run_full_pipeline.py reduce_scatter --no-viz

    # Only run analysis (no benchmark)
    python3 run_full_pipeline.py all_reduce --analyze-only --run-dir /path/to/run_dir
"""

import argparse
import os
import sys
import subprocess
import glob
import time
from pathlib import Path


class PipelineRunner:
    """Orchestrate the complete RCCL analysis pipeline."""
    
    def __init__(self, args):
        self.args = args
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.run_dir = args.run_dir
        self.benchmark_name = args.benchmark
        self.success_count = 0
        self.fail_count = 0
        self.outputs = []
        
    def run_command(self, script_name, run_dir=None, description=None):
        """Run a pipeline script and track results."""
        script_path = os.path.join(self.script_dir, script_name)
        
        if not os.path.exists(script_path):
            print(f"❌ Error: Script not found: {script_path}")
            self.fail_count += 1
            return False
        
        cmd = ['python3', script_path]
        
        # Add run_dir if provided
        if run_dir:
            cmd.append(run_dir)
        
        desc = description or script_name
        print(f"\n{'='*80}")
        print(f"Running: {desc}")
        print(f"Command: {' '.join(cmd)}")
        print('='*80)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=False,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                print(f"✅ {desc} completed successfully")
                self.success_count += 1
                return True
            else:
                print(f"❌ {desc} failed with exit code {result.returncode}")
                self.fail_count += 1
                return False
                
        except Exception as e:
            print(f"❌ {desc} failed with exception: {e}")
            self.fail_count += 1
            return False
    
    def find_latest_run_dir(self):
        """Find the most recently created run directory for this benchmark."""
        import socket
        hostname = socket.gethostname()
        data_dir = f"/work/lmeadows/rccl/data/{hostname}"
        
        if not os.path.exists(data_dir):
            print(f"❌ Error: Data directory not found: {data_dir}")
            return None
        
        pattern = os.path.join(data_dir, f"run_{self.benchmark_name}_*")
        run_dirs = glob.glob(pattern)
        
        if not run_dirs:
            print(f"❌ Error: No run directories found matching: {pattern}")
            return None
        
        # Sort by modification time, most recent first
        run_dirs.sort(key=os.path.getmtime, reverse=True)
        latest = run_dirs[0]
        
        print(f"\n✅ Found run directory: {latest}")
        return latest
    
    def scan_outputs(self):
        """Scan the run directory for generated outputs."""
        if not self.run_dir or not os.path.exists(self.run_dir):
            return
        
        output_patterns = [
            ('Timing CSVs', 'all_rank*.csv'),
            ('Benchmark Output', f'{self.benchmark_name}_benchmark_output.txt'),
            ('Statistical Analysis', f'{self.benchmark_name}_timing_analysis.csv'),
            ('BIC Segmentation', f'{self.benchmark_name}_bic_segmentation.json'),
            ('Interactive Plot', f'{self.benchmark_name}_size_vs_time.html'),
            ('Boxplots', f'{self.benchmark_name}_segment*_boxplots.png'),
            ('Metadata', 'run_metadata.json'),
        ]
        
        print(f"\n{'='*80}")
        print("Generated Outputs")
        print('='*80)
        
        for name, pattern in output_patterns:
            matches = glob.glob(os.path.join(self.run_dir, pattern))
            if matches:
                print(f"\n{name}:")
                for match in sorted(matches):
                    size = os.path.getsize(match)
                    size_str = self.format_size(size)
                    rel_path = os.path.relpath(match, self.run_dir)
                    print(f"  ✓ {rel_path} ({size_str})")
                    self.outputs.append(match)
            else:
                print(f"\n{name}:")
                print(f"  ✗ Not found")
    
    def format_size(self, size_bytes):
        """Format file size in human-readable form."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def run_benchmark(self):
        """Step 1: Run benchmark with timing sweep."""
        if self.args.analyze_only:
            print("\n⏭️  Skipping benchmark run (--analyze-only)")
            return True
        
        script_path = os.path.join(self.script_dir, 'run_timing_sweep.py')
        
        cmd = ['python3', script_path, self.benchmark_name]
        
        # Add optional arguments
        if self.args.ranks:
            cmd.extend(['--ranks', str(self.args.ranks)])
        if self.args.iterations:
            cmd.extend(['--iterations', str(self.args.iterations)])
        if self.args.min_size:
            cmd.extend(['--min-size', str(self.args.min_size)])
        if self.args.max_size:
            cmd.extend(['--max-size', str(self.args.max_size)])
        if self.args.warmup:
            cmd.extend(['--warmup', str(self.args.warmup)])
        
        print(f"\n{'='*80}")
        print("Step 1: Running Benchmark with Timing Sweep")
        print(f"Command: {' '.join(cmd)}")
        print('='*80)
        
        try:
            result = subprocess.run(cmd, check=False)
            
            if result.returncode == 0:
                print(f"✅ Benchmark completed successfully")
                self.success_count += 1
                return True
            else:
                print(f"❌ Benchmark failed with exit code {result.returncode}")
                self.fail_count += 1
                return False
                
        except Exception as e:
            print(f"❌ Benchmark failed with exception: {e}")
            self.fail_count += 1
            return False
    
    def run_correlation(self):
        """Step 2: Correlate ROCProfiler kernel traces (automatic in run_timing_sweep)."""
        # This step is now automatic in run_timing_sweep.py, but we can verify
        # Check if all_rank*.csv files exist
        csv_files = glob.glob(os.path.join(self.run_dir, 'all_rank*.csv'))
        
        if csv_files:
            print(f"\n✅ Correlation already completed ({len(csv_files)} rank files found)")
            return True
        else:
            print(f"\n⚠️  Warning: No timing CSV files found, attempting manual correlation...")
            return self.run_command(
                'correlate_rocprof_timings.py',
                self.run_dir,
                'Step 2: Correlate ROCProfiler Traces'
            )
    
    def run_analysis(self):
        """Step 3: Generate statistical analysis."""
        return self.run_command(
            'analyze_timing_stats.py',
            self.run_dir,
            'Step 3: Statistical Analysis'
        )
    
    def run_segmentation(self):
        """Step 4: Perform BIC segmentation."""
        return self.run_command(
            'segment_performance_bic.py',
            self.run_dir,
            'Step 4: BIC Segmentation'
        )
    
    def run_visualizations(self):
        """Steps 5-6: Create all visualizations."""
        if self.args.no_viz:
            print("\n⏭️  Skipping visualizations (--no-viz)")
            return True
        
        success = True
        
        # Interactive plot
        if not self.run_command(
            'plot_size_vs_time_plotly.py',
            self.run_dir,
            'Step 5: Interactive Size vs Time Plot'
        ):
            success = False
        
        # Boxplots
        if not self.run_command(
            'create_boxplots.py',
            self.run_dir,
            'Step 6: Boxplot Visualizations'
        ):
            success = False
        
        # Note: plot_kernel_timeline.py removed - needs optimization
        # Can be run manually if needed: python3 plot_kernel_timeline.py <run_dir>
        
        return success
    
    def print_summary(self):
        """Print final summary."""
        print(f"\n{'='*80}")
        print("Pipeline Execution Summary")
        print('='*80)
        print(f"Benchmark: {self.benchmark_name}")
        print(f"Run Directory: {self.run_dir}")
        print(f"\nResults:")
        print(f"  ✅ Successful steps: {self.success_count}")
        print(f"  ❌ Failed steps: {self.fail_count}")
        print(f"  📁 Output files: {len(self.outputs)}")
        
        if self.fail_count == 0:
            print(f"\n🎉 Pipeline completed successfully!")
            print(f"\nView results:")
            print(f"  cd {self.run_dir}")
            
            # Find HTML files
            html_files = glob.glob(os.path.join(self.run_dir, '*.html'))
            if html_files:
                print(f"\n  Open interactive visualizations:")
                for html in html_files:
                    print(f"    firefox {html}")
        else:
            print(f"\n⚠️  Pipeline completed with {self.fail_count} failed step(s)")
        
        print('='*80)
    
    def run(self):
        """Execute the complete pipeline."""
        start_time = time.time()
        
        print("\n" + "="*80)
        print("RCCL Performance Analysis Pipeline")
        print("="*80)
        print(f"Benchmark: {self.benchmark_name}")
        if not self.args.analyze_only:
            print(f"Ranks: {self.args.ranks}")
            print(f"Iterations: {self.args.iterations}")
        print("="*80)
        
        # Step 1: Run benchmark (unless --analyze-only)
        if not self.args.analyze_only:
            if not self.run_benchmark():
                print("\n❌ Benchmark failed, stopping pipeline")
                return 1
            
            # Find the run directory
            time.sleep(1)  # Give filesystem time to update
            self.run_dir = self.find_latest_run_dir()
            if not self.run_dir:
                return 1
        else:
            # Use provided run_dir
            if not self.run_dir:
                print("❌ Error: --analyze-only requires --run-dir")
                return 1
            if not os.path.exists(self.run_dir):
                print(f"❌ Error: Run directory not found: {self.run_dir}")
                return 1
        
        # Step 2: Verify correlation (usually automatic)
        self.run_correlation()
        
        # Step 3: Statistical analysis
        if not self.run_analysis():
            if not self.args.continue_on_error:
                print("\n❌ Analysis failed, stopping pipeline")
                return 1
        
        # Step 4: BIC segmentation
        if not self.run_segmentation():
            if not self.args.continue_on_error:
                print("\n⚠️  Segmentation failed, continuing without segment data")
        
        # Steps 5-6: Visualizations
        self.run_visualizations()
        
        # Scan outputs
        self.scan_outputs()
        
        # Print summary
        elapsed = time.time() - start_time
        print(f"\nTotal execution time: {elapsed:.1f} seconds")
        self.print_summary()
        
        return 0 if self.fail_count == 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description='Run complete RCCL performance analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline for all_reduce with 8 ranks
  %(prog)s all_reduce

  # Custom configuration
  %(prog)s all_reduce --ranks 4 --iterations 50 --min-size 1024

  # Only analyze existing data (no benchmark)
  %(prog)s all_reduce --analyze-only --run-dir /path/to/run_all_reduce_*

  # Skip visualizations (faster)
  %(prog)s reduce_scatter --no-viz

Available benchmarks:
  all_reduce, all_gather, broadcast, reduce, reduce_scatter,
  scatter, gather, alltoall, alltoallv, hypercube, sendrecv
        """
    )
    
    # Required arguments
    parser.add_argument('benchmark',
                        help='Benchmark name (e.g., all_reduce, all_gather)')
    
    # Benchmark configuration
    parser.add_argument('--ranks', type=int, default=8,
                        help='Number of MPI ranks (default: 8)')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of timed iterations (default: 100)')
    parser.add_argument('--warmup', type=int, default=5,
                        help='Number of warmup iterations (default: 5)')
    parser.add_argument('--min-size', type=str,
                        help='Minimum message size (default: auto-adjusted)')
    parser.add_argument('--max-size', type=str, default='1G',
                        help='Maximum message size (default: 1G)')
    
    # Pipeline control
    parser.add_argument('--analyze-only', action='store_true',
                        help='Skip benchmark run, only analyze existing data')
    parser.add_argument('--run-dir', type=str,
                        help='Run directory (required with --analyze-only)')
    parser.add_argument('--no-viz', action='store_true',
                        help='Skip visualization steps')
    parser.add_argument('--continue-on-error', action='store_true',
                        help='Continue pipeline even if a step fails')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.analyze_only and not args.run_dir:
        parser.error("--analyze-only requires --run-dir")
    
    # Run pipeline
    runner = PipelineRunner(args)
    return runner.run()


if __name__ == '__main__':
    sys.exit(main())

