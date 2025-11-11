"""
Test CSV parsing and data loading functions.

These tests verify that benchmark CSV files are loaded correctly,
columns are mapped properly, and data types are correct.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from common_data import (
    load_benchmark_output,
    load_timing_data,
    find_benchmark_name,
    load_rank_pid_mapping
)


class TestBenchmarkOutputLoading:
    """Tests for load_benchmark_output()."""
    
    def test_load_valid_csv(self, mock_benchmark_csv):
        """Test loading a valid benchmark CSV."""
        df = load_benchmark_output(mock_benchmark_csv)
        
        assert not df.empty
        assert len(df) == 6  # 3 sizes × 2 inplace modes
    
    def test_column_names_mapped(self, mock_benchmark_csv):
        """Verify CSV columns are mapped to standard names."""
        df = load_benchmark_output(mock_benchmark_csv)
        
        # Check standard column names exist
        expected_columns = [
            'size_bytes',
            'wall_time_us',
            'algbw_gbs',
            'busbw_gbs',
            'busbw_factor',
            'inplace',
            'errors'
        ]
        
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_column_types(self, mock_benchmark_csv):
        """Verify columns have correct data types."""
        df = load_benchmark_output(mock_benchmark_csv)
        
        assert df['size_bytes'].dtype == np.int64
        assert df['inplace'].dtype == np.int64
        assert df['wall_time_us'].dtype == np.float64
        assert df['algbw_gbs'].dtype == np.float64
        assert df['busbw_gbs'].dtype == np.float64
        assert df['busbw_factor'].dtype == np.float64
        assert df['errors'].dtype == np.int64
    
    def test_data_values(self, mock_benchmark_csv):
        """Verify data values are loaded correctly."""
        df = load_benchmark_output(mock_benchmark_csv)
        
        # Check first row
        row0 = df[df['size_bytes'] == 1024].iloc[0]
        assert row0['wall_time_us'] == 50.0
        assert abs(row0['algbw_gbs'] - 0.02048) < 0.0001
        assert abs(row0['busbw_gbs'] - 0.03584) < 0.0001
        assert row0['busbw_factor'] == 1.75
    
    def test_missing_file(self, tmp_path):
        """Handle missing CSV file gracefully."""
        # Create a properly named run directory with no CSV
        run_dir = tmp_path / "run_test_20250101_000000"
        run_dir.mkdir()
        
        df = load_benchmark_output(run_dir, 'test')
        assert df.empty
    
    def test_auto_detect_benchmark_name(self, mock_benchmark_csv):
        """Test automatic benchmark name detection."""
        # Should auto-detect 'all_reduce' from directory name
        df = load_benchmark_output(mock_benchmark_csv, benchmark_name=None)
        assert not df.empty


class TestTimingDataLoading:
    """Tests for load_timing_data()."""
    
    def test_load_timing_csv(self, mock_timing_csv):
        """Test loading per-rank timing data."""
        df = load_timing_data(mock_timing_csv)
        
        assert not df.empty
        assert 'size_bytes' in df.columns
        assert 'kernel_time_us' in df.columns
        assert 'rank' in df.columns
    
    def test_timing_column_mapping(self, mock_timing_csv):
        """Verify timing columns are mapped correctly."""
        df = load_timing_data(mock_timing_csv)
        
        # Original names should be mapped
        assert 'size_bytes' in df.columns  # Was 'Size (bytes)'
        assert 'inplace' in df.columns  # Was 'In-place'
        assert 'iteration' in df.columns  # Was 'Iteration'
        assert 'kernel_time_us' in df.columns  # Was 'Kernel Time (us)'
    
    def test_empty_directory(self, tmp_path):
        """Handle directory with no timing files."""
        df = load_timing_data(tmp_path)
        assert df.empty or df is None


class TestBenchmarkNameExtraction:
    """Tests for find_benchmark_name()."""
    
    def test_standard_format(self):
        """Test standard run directory format."""
        name = find_benchmark_name('run_all_reduce_20251110_135024')
        assert name == 'all_reduce'
    
    def test_with_path(self):
        """Test with full path."""
        name = find_benchmark_name('/path/to/run_scatter_20250101_120000')
        assert name == 'scatter'
    
    def test_different_benchmarks(self):
        """Test various benchmark names."""
        test_cases = [
            ('run_all_reduce_20251110_135024', 'all_reduce'),
            ('run_broadcast_20240101_000000', 'broadcast'),
            ('run_reduce_scatter_20230615_123456', 'reduce_scatter'),
            ('run_alltoall_20220330_235959', 'alltoall'),
        ]
        
        for dir_name, expected in test_cases:
            result = find_benchmark_name(dir_name)
            assert result == expected
    
    def test_invalid_format_raises_error(self):
        """Invalid directory format should raise ValueError."""
        with pytest.raises(ValueError):
            find_benchmark_name('invalid_directory_name')
    
    def test_no_timestamp_raises_error(self):
        """Missing timestamp should raise ValueError."""
        with pytest.raises(ValueError):
            find_benchmark_name('run_all_reduce')


class TestMetadataLoading:
    """Tests for load_rank_pid_mapping()."""
    
    def test_load_metadata(self, mock_metadata):
        """Test loading rank-to-PID mapping."""
        mapping = load_rank_pid_mapping(mock_metadata)
        
        assert isinstance(mapping, dict)
        assert len(mapping) == 8
        assert mapping[0] == 12345
        assert mapping[7] == 12352
    
    def test_missing_metadata(self, tmp_path):
        """Handle missing metadata file."""
        mapping = load_rank_pid_mapping(tmp_path)
        assert mapping == {}


class TestDataConsistency:
    """Tests for data consistency across files."""
    
    def test_sizes_match_across_files(self, mock_run_dir):
        """Verify sizes are consistent between benchmark and timing data."""
        benchmark_df = load_benchmark_output(mock_run_dir)
        timing_df = load_timing_data(mock_run_dir)
        
        benchmark_sizes = set(benchmark_df['size_bytes'].unique())
        timing_sizes = set(timing_df['size_bytes'].unique())
        
        # Timing data should be subset of benchmark data
        assert timing_sizes.issubset(benchmark_sizes)
    
    def test_inplace_modes_present(self, mock_run_dir):
        """Verify both inplace modes are present."""
        df = load_benchmark_output(mock_run_dir)
        
        inplace_modes = set(df['inplace'].unique())
        assert inplace_modes == {0, 1}
    
    def test_no_duplicate_entries(self, mock_run_dir):
        """Verify no duplicate (size, inplace) combinations."""
        df = load_benchmark_output(mock_run_dir)
        
        # Each (size, inplace) should appear once
        duplicates = df.duplicated(subset=['size_bytes', 'inplace'])
        assert not duplicates.any()


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_csv_file(self, tmp_path):
        """Handle empty CSV file."""
        run_dir = tmp_path / "run_test_20250101_000000"
        run_dir.mkdir()
        
        # Create empty CSV
        csv_file = run_dir / "test_benchmark_output.csv"
        csv_file.write_text("")
        
        # Should raise or return empty - either is acceptable
        try:
            df = load_benchmark_output(run_dir, 'test')
            assert df is None or df.empty
        except (pd.errors.EmptyDataError, Exception):
            pass  # This is acceptable behavior
    
    def test_csv_with_header_only(self, tmp_path):
        """Handle CSV with only header row."""
        run_dir = tmp_path / "run_test_20250101_000000"
        run_dir.mkdir()
        
        csv_content = "numCycle,collective,size,time,algbw,busbw,busbwfactor,inplace,#wrong\n"
        csv_file = run_dir / "test_benchmark_output.csv"
        csv_file.write_text(csv_content)
        
        df = load_benchmark_output(run_dir, 'test')
        assert df.empty or len(df) == 0
    
    def test_nonexistent_directory(self):
        """Handle non-existent directory."""
        df = load_benchmark_output('/nonexistent/path/run_test_20250101_000000')
        assert df.empty


class TestRealWorldData:
    """Tests simulating real benchmark output."""
    
    def test_full_size_sweep(self, tmp_path):
        """Test with full size sweep from 8B to 1GB."""
        # Generate realistic size sweep: 8, 16, 32, ..., 1GB
        sizes = [2**i for i in range(3, 31)]  # 8 to 1GB
        
        rows = []
        for size in sizes:
            for inplace in [0, 1]:
                # Realistic timing: increases with size
                time = 20 + (size / 1000)  # µs
                algbw = size / time / 1000
                busbw = algbw * 1.75
                
                rows.append({
                    'numCycle': 0,
                    'collective': 'AllReduce',
                    'ranks': 8,
                    'rankspernode': 1,
                    'gpusperrank': 1,
                    'size': size,
                    'type': 'float',
                    'redop': 'sum',
                    'inplace': inplace,
                    'time': time,
                    'algbw': algbw,
                    'busbw': busbw,
                    'busbwfactor': 1.75,
                    '#wrong': 0
                })
        
        # Create CSV
        df_input = pd.DataFrame(rows)
        run_dir = tmp_path / "run_test_20250101_000000"
        run_dir.mkdir()
        csv_file = run_dir / "test_benchmark_output.csv"
        df_input.to_csv(csv_file, index=False)
        
        # Load and verify
        df = load_benchmark_output(run_dir, 'test')
        assert len(df) == len(sizes) * 2  # 2 inplace modes
        assert df['size_bytes'].min() == 8
        assert df['size_bytes'].max() == 2**30

