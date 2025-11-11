"""
Pytest fixtures for RCCL benchmark tests.
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
from pathlib import Path


@pytest.fixture
def mock_benchmark_csv(tmp_path):
    """Create a minimal mock benchmark CSV with correct format."""
    # Bandwidth values: algbw = size / time / 1000, busbw = algbw * 1.75
    csv_content = """numCycle,collective,ranks,rankspernode,gpusperrank,size,type,redop,inplace,time,algbw,busbw,busbwfactor,#wrong
0,AllReduce,8,1,1,1024,float,sum,0,50.0,0.02048,0.03584,1.75,0
0,AllReduce,8,1,1,1024,float,sum,1,51.0,0.02008,0.03514,1.75,0
0,AllReduce,8,1,1,2048,float,sum,0,60.0,0.03413,0.05973,1.75,0
0,AllReduce,8,1,1,2048,float,sum,1,61.0,0.03357,0.05875,1.75,0
0,AllReduce,8,1,1,4096,float,sum,0,80.0,0.0512,0.0896,1.75,0
0,AllReduce,8,1,1,4096,float,sum,1,82.0,0.04995,0.08741,1.75,0
"""
    csv_file = tmp_path / "all_reduce_benchmark_output.csv"
    csv_file.write_text(csv_content)
    
    # Create a mock run directory structure
    run_dir = tmp_path / "run_all_reduce_20251110_135024"
    run_dir.mkdir()
    (run_dir / "all_reduce_benchmark_output.csv").write_text(csv_content)
    
    return run_dir


@pytest.fixture
def mock_timing_csv(tmp_path):
    """Create mock per-rank timing data."""
    # Create timing data with some variance
    np.random.seed(42)
    
    sizes = [1024, 2048, 4096]
    iterations = 20
    
    data = []
    for size in sizes:
        for inplace in [0, 1]:
            for iteration in range(iterations):
                # Add some realistic variance
                base_time = size / 20.0  # Roughly proportional to size
                noise = np.random.normal(0, base_time * 0.05)
                time = base_time + noise
                
                data.append({
                    'Size (bytes)': size,
                    'In-place': inplace,
                    'Iteration': iteration,
                    'Kernel Time (us)': max(time, 1.0),  # Ensure positive
                    'IQR Filtered': True
                })
    
    df = pd.DataFrame(data)
    
    run_dir = tmp_path / "run_all_reduce_20251110_135024"
    run_dir.mkdir(exist_ok=True)
    
    csv_file = run_dir / "all_rank0.csv"
    df.to_csv(csv_file, index=False)
    
    return run_dir


@pytest.fixture
def mock_run_dir(mock_benchmark_csv, mock_timing_csv):
    """Full mock run directory with all necessary files."""
    # They both create the same directory, so just return one
    return mock_benchmark_csv


@pytest.fixture
def mock_metadata(tmp_path):
    """Create mock run metadata."""
    metadata = {
        "benchmark": "all_reduce",
        "timestamp": "20251110_135024",
        "hostname": "test-node",
        "ranks": 8,
        "min_size": 8,
        "max_size": 1073741824,
        "iterations": 100,
        "warmup": 5,
        "rank_pid_mapping": {
            "0": 12345,
            "1": 12346,
            "2": 12347,
            "3": 12348,
            "4": 12349,
            "5": 12350,
            "6": 12351,
            "7": 12352
        }
    }
    
    run_dir = tmp_path / "run_all_reduce_20251110_135024"
    run_dir.mkdir(exist_ok=True)
    
    metadata_file = run_dir / "run_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return run_dir


@pytest.fixture
def sample_sizes():
    """Sample message sizes for testing."""
    return np.array([1024, 2048, 4096, 8192, 16384])


@pytest.fixture
def sample_times():
    """Sample timing values for testing."""
    return np.array([50.0, 60.0, 80.0, 110.0, 150.0])

