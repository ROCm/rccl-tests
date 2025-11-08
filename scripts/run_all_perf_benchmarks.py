#!/usr/bin/env python3
"""
Run pipeline on all RCCL performance benchmarks.
"""

import subprocess
import sys
import os

BENCHMARKS = [
    "all_reduce",
    "all_gather",
    "broadcast",
    "reduce",
    "reduce_scatter",
    "scatter",
    "gather",
    "alltoall",
    "alltoallv",
    "hypercube",
    "sendrecv",
    "all_reduce_bias",
]

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_script = os.path.join(script_dir, 'run_full_pipeline.py')
    
    print("="*80)
    print(f"Running pipeline on {len(BENCHMARKS)} benchmarks")
    print("="*80)
    print()
    
    success_count = 0
    fail_count = 0
    failed_benchmarks = []
    
    for i, benchmark in enumerate(BENCHMARKS, 1):
        print("="*80)
        print(f"Starting: {benchmark} ({i}/{len(BENCHMARKS)})")
        print("="*80)
        
        try:
            result = subprocess.run(
                ['python3', pipeline_script, benchmark],
                check=False
            )
            
            if result.returncode == 0:
                print(f"✅ {benchmark} completed successfully")
                success_count += 1
            else:
                print(f"❌ {benchmark} failed with exit code {result.returncode}")
                fail_count += 1
                failed_benchmarks.append(benchmark)
        
        except Exception as e:
            print(f"❌ {benchmark} failed with exception: {e}")
            fail_count += 1
            failed_benchmarks.append(benchmark)
        
        print()
        print(f"Progress: {success_count + fail_count} / {len(BENCHMARKS)} completed")
        print()
    
    print("="*80)
    print("Final Summary")
    print("="*80)
    print(f"Total benchmarks: {len(BENCHMARKS)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    
    if failed_benchmarks:
        print()
        print("Failed benchmarks:")
        for benchmark in failed_benchmarks:
            print(f"  - {benchmark}")
    
    print("="*80)
    
    return fail_count

if __name__ == '__main__':
    sys.exit(main())


