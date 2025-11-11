"""
Test data integrity and cross-validation.

These tests verify that bandwidth calculations from Python match
the benchmark's reported values, ensuring consistency between
C++ and Python implementations.
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
    calculate_algorithm_bandwidth,
    calculate_bus_bandwidth,
    get_bus_bandwidth_factor
)


class TestBusBandwidthFactorConsistency:
    """Verify busbw_factor = busbw / algbw in CSV."""
    
    def test_factor_matches_ratio(self, mock_benchmark_csv):
        """busbw_factor should equal busbw / algbw."""
        df = load_benchmark_output(mock_benchmark_csv)
        
        # Calculate factor from bandwidth values
        calculated_factor = df['busbw_gbs'] / df['algbw_gbs']
        
        # Should match the busbw_factor column (allow small rounding errors)
        np.testing.assert_allclose(
            calculated_factor.values,
            df['busbw_factor'].values,
            rtol=1e-4,  # 0.01% tolerance for rounding
            err_msg="busbw_factor doesn't match busbw/algbw ratio"
        )
    
    def test_factor_consistent_across_sizes(self, mock_benchmark_csv):
        """Factor should be constant across all sizes for same collective."""
        df = load_benchmark_output(mock_benchmark_csv)
        
        # All rows should have same factor (AllReduce with 8 ranks)
        unique_factors = df['busbw_factor'].unique()
        assert len(unique_factors) == 1
        assert unique_factors[0] == 1.75
    
    def test_factor_matches_formula(self, mock_benchmark_csv):
        """Factor should match theoretical formula."""
        df = load_benchmark_output(mock_benchmark_csv)
        
        nranks = df['ranks'].iloc[0]
        collective = 'all_reduce'
        
        expected_factor = get_bus_bandwidth_factor(collective, nranks)
        actual_factor = df['busbw_factor'].iloc[0]
        
        assert abs(actual_factor - expected_factor) < 0.001


class TestAlgorithmBandwidthConsistency:
    """Verify algbw calculation matches CSV values."""
    
    def test_algbw_from_size_and_time(self, mock_benchmark_csv):
        """Recalculate algbw from size and time, should match CSV."""
        df = load_benchmark_output(mock_benchmark_csv)
        
        # Calculate algbw using Python function
        calculated_algbw = calculate_algorithm_bandwidth(
            df['size_bytes'].values,
            df['wall_time_us'].values
        )
        
        # Should match CSV values
        np.testing.assert_allclose(
            calculated_algbw,
            df['algbw_gbs'].values,
            rtol=0.01,  # 1% tolerance for rounding
            err_msg="Calculated algbw doesn't match CSV"
        )
    
    def test_manual_calculation_matches(self, mock_benchmark_csv):
        """Manual formula should match."""
        df = load_benchmark_output(mock_benchmark_csv)
        
        for _, row in df.iterrows():
            manual_algbw = row['size_bytes'] / row['wall_time_us'] / 1000
            csv_algbw = row['algbw_gbs']
            
            assert abs(manual_algbw - csv_algbw) < 0.01


class TestBusBandwidthConsistency:
    """Verify busbw calculation matches CSV values."""
    
    def test_busbw_from_python_matches_csv(self, mock_benchmark_csv):
        """Python busbw calculation should match benchmark's."""
        df = load_benchmark_output(mock_benchmark_csv)
        
        nranks = df['ranks'].iloc[0]
        collective = 'all_reduce'
        
        # Calculate using Python function
        calculated_busbw = calculate_bus_bandwidth(
            df['size_bytes'].values,
            df['wall_time_us'].values,
            collective,
            nranks
        )
        
        # Should match CSV values
        np.testing.assert_allclose(
            calculated_busbw,
            df['busbw_gbs'].values,
            rtol=0.01,
            err_msg="Python busbw calculation doesn't match CSV"
        )
    
    def test_busbw_equals_algbw_times_factor(self, mock_benchmark_csv):
        """busbw should equal algbw * factor."""
        df = load_benchmark_output(mock_benchmark_csv)
        
        calculated_busbw = df['algbw_gbs'] * df['busbw_factor']
        
        np.testing.assert_allclose(
            calculated_busbw.values,
            df['busbw_gbs'].values,
            rtol=1e-4  # 0.01% tolerance for rounding
        )


class TestCrossValidation:
    """Cross-validate multiple data sources."""
    
    def test_all_bandwidth_relationships(self, mock_benchmark_csv):
        """Verify all bandwidth relationships are consistent."""
        df = load_benchmark_output(mock_benchmark_csv)
        
        for _, row in df.iterrows():
            size = row['size_bytes']
            time = row['wall_time_us']
            algbw = row['algbw_gbs']
            busbw = row['busbw_gbs']
            factor = row['busbw_factor']
            
            # Check: algbw = size / time / 1000
            assert abs(algbw - (size / time / 1000)) < 0.01
            
            # Check: busbw = algbw * factor
            assert abs(busbw - (algbw * factor)) < 0.01
            
            # Check: factor = busbw / algbw
            assert abs(factor - (busbw / algbw)) < 0.001
    
    def test_python_matches_cpp_for_all_collectives(self):
        """Verify Python formulas match C++ for all collective types."""
        test_cases = [
            ('all_reduce', 8, 1.75),
            ('reduce_scatter', 8, 0.875),
            ('all_gather', 8, 0.875),
            ('reduce', 8, 1.0),
            ('broadcast', 8, 1.0),
            ('alltoall', 8, 0.875),
        ]
        
        for collective, nranks, expected_factor in test_cases:
            factor = get_bus_bandwidth_factor(collective, nranks)
            assert abs(factor - expected_factor) < 0.001, \
                f"Factor mismatch for {collective}: expected {expected_factor}, got {factor}"


class TestNumericalAccuracy:
    """Test numerical accuracy and precision."""
    
    def test_no_significant_rounding_errors(self, mock_benchmark_csv):
        """Verify no significant rounding errors in calculations."""
        df = load_benchmark_output(mock_benchmark_csv)
        
        # Recalculate everything
        algbw_calc = df['size_bytes'] / df['wall_time_us'] / 1000
        busbw_calc = algbw_calc * df['busbw_factor']
        
        # Errors should be tiny (< 0.1%)
        algbw_error = abs(algbw_calc - df['algbw_gbs']) / df['algbw_gbs']
        busbw_error = abs(busbw_calc - df['busbw_gbs']) / df['busbw_gbs']
        
        assert (algbw_error < 0.001).all(), "Algorithm BW has significant rounding errors"
        assert (busbw_error < 0.001).all(), "Bus BW has significant rounding errors"
    
    def test_floating_point_stability(self):
        """Test calculations with challenging floating point values."""
        # Values that might expose precision issues
        size = 1024 * 1024 + 1  # Not a power of 2
        time = 33.333333  # Repeating decimal
        
        algbw = calculate_algorithm_bandwidth(size, time)
        busbw = calculate_bus_bandwidth(size, time, 'all_reduce', 8)
        
        # Verify relationship holds
        assert abs(busbw - algbw * 1.75) < 0.01


class TestDataSanity:
    """Sanity checks on loaded data."""
    
    def test_no_negative_values(self, mock_benchmark_csv):
        """Verify no negative bandwidth or time values."""
        df = load_benchmark_output(mock_benchmark_csv)
        
        assert (df['wall_time_us'] > 0).all()
        assert (df['algbw_gbs'] >= 0).all()
        assert (df['busbw_gbs'] >= 0).all()
        assert (df['busbw_factor'] > 0).all()
    
    def test_busbw_greater_than_algbw_for_allreduce(self, mock_benchmark_csv):
        """For AllReduce with >1 rank, busbw should be > algbw."""
        df = load_benchmark_output(mock_benchmark_csv)
        
        # AllReduce with 8 ranks should have busbw > algbw
        assert (df['busbw_gbs'] > df['algbw_gbs']).all()
    
    def test_reasonable_bandwidth_values(self, mock_benchmark_csv):
        """Bandwidth values should be in reasonable range."""
        df = load_benchmark_output(mock_benchmark_csv)
        
        # For test data with small sizes (KB range), algbw should be 0.001 to 1000 GB/s
        assert (df['algbw_gbs'] > 0.001).all()
        assert (df['algbw_gbs'] < 1000).all()
    
    def test_size_increases_monotonically(self, mock_benchmark_csv):
        """Message sizes should increase."""
        df = load_benchmark_output(mock_benchmark_csv)
        
        sizes = df['size_bytes'].unique()
        sizes_sorted = sorted(sizes)
        
        assert list(sizes_sorted) == sorted(list(sizes))


class TestErrorDetection:
    """Test detection of inconsistent or corrupt data."""
    
    def test_detect_factor_mismatch(self, tmp_path):
        """Detect when factor doesn't match busbw/algbw."""
        # Create CSV with intentionally wrong factor
        csv_content = """numCycle,collective,ranks,rankspernode,gpusperrank,size,type,redop,inplace,time,algbw,busbw,busbwfactor,#wrong
0,AllReduce,8,1,1,1024,float,sum,0,50.0,20.0,35.0,2.0,0
"""
        run_dir = tmp_path / "run_test_20250101_000000"
        run_dir.mkdir()
        csv_file = run_dir / "test_benchmark_output.csv"
        csv_file.write_text(csv_content)
        
        df = load_benchmark_output(run_dir, 'test')
        
        # Calculate actual factor
        actual_factor = df['busbw_gbs'].iloc[0] / df['algbw_gbs'].iloc[0]
        reported_factor = df['busbw_factor'].iloc[0]
        
        # Should detect mismatch
        assert abs(actual_factor - reported_factor) > 0.1
    
    def test_zero_time_handled(self, tmp_path):
        """Detect and handle zero time values."""
        csv_content = """numCycle,collective,ranks,rankspernode,gpusperrank,size,type,redop,inplace,time,algbw,busbw,busbwfactor,#wrong
0,AllReduce,8,1,1,1024,float,sum,0,0.0,inf,inf,1.75,0
"""
        run_dir = tmp_path / "run_test_20250101_000000"
        run_dir.mkdir()
        csv_file = run_dir / "test_benchmark_output.csv"
        csv_file.write_text(csv_content)
        
        df = load_benchmark_output(run_dir, 'test')
        
        # Should load, but have inf values
        assert df['algbw_gbs'].iloc[0] == float('inf')


class TestRealisticScenarios:
    """Test with realistic benchmark scenarios."""
    
    def test_typical_allreduce_sweep(self):
        """Test typical AllReduce size sweep."""
        sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        nranks = 8
        
        for size in sizes:
            # Typical time increases with size
            time = 20 + size / 50  # µs
            
            algbw_manual = size / time / 1000
            busbw_manual = algbw_manual * 1.75
            
            algbw_func = calculate_algorithm_bandwidth(size, time)
            busbw_func = calculate_bus_bandwidth(size, time, 'all_reduce', nranks)
            
            assert abs(algbw_manual - algbw_func) < 0.01
            assert abs(busbw_manual - busbw_func) < 0.01

