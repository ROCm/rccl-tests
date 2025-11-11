"""
Test bandwidth calculation functions.

These tests verify the correctness of algorithm and bus bandwidth calculations,
which are critical for accurate performance analysis.
"""
import pytest
import numpy as np
import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from common_data import (
    calculate_algorithm_bandwidth,
    calculate_bus_bandwidth,
    get_bus_bandwidth_factor
)


class TestAlgorithmBandwidth:
    """Tests for algorithm bandwidth calculation."""
    
    def test_basic_calculation(self):
        """Verify algbw = size / time / 1000"""
        # 1000 bytes in 50 µs = 0.02 GB/s
        result = calculate_algorithm_bandwidth(1000, 50)
        assert result == 0.02
    
    def test_large_message(self):
        """Test with large message size."""
        # 1 GB in 1000 µs = 1 GB/s (1e9 bytes / 1000 µs / 1000 = 1000)
        result = calculate_algorithm_bandwidth(1_000_000_000, 1000)
        assert result == 1000.0
    
    def test_small_message(self):
        """Test with small message size."""
        # 10 bytes in 100 µs = 0.0001 GB/s
        result = calculate_algorithm_bandwidth(10, 100)
        assert result == 0.0001
    
    def test_vectorized_calculation(self):
        """Test with numpy arrays."""
        sizes = np.array([1000, 2000, 3000])
        times = np.array([50, 100, 150])
        result = calculate_algorithm_bandwidth(sizes, times)
        expected = np.array([0.02, 0.02, 0.02])
        np.testing.assert_array_equal(result, expected)
    
    def test_zero_time_handling(self):
        """Verify behavior with zero time (should raise or be inf)."""
        with pytest.raises(ZeroDivisionError):
            result = calculate_algorithm_bandwidth(1000, 0)
    
    def test_different_sizes_times(self):
        """Test various size/time combinations."""
        test_cases = [
            (1024, 50, 0.02048),
            (2048, 60, 0.034133333),
            (4096, 80, 0.0512),
        ]
        for size, time, expected in test_cases:
            result = calculate_algorithm_bandwidth(size, time)
            assert abs(result - expected) < 0.0001


class TestBusBandwidthFactor:
    """Tests for bus bandwidth factor by collective type."""
    
    def test_allreduce_8_ranks(self):
        """AllReduce with 8 ranks: factor = 2*(8-1)/8 = 1.75"""
        factor = get_bus_bandwidth_factor('all_reduce', 8)
        assert factor == 1.75
    
    def test_allreduce_4_ranks(self):
        """AllReduce with 4 ranks: factor = 2*(4-1)/4 = 1.5"""
        factor = get_bus_bandwidth_factor('all_reduce', 4)
        assert factor == 1.5
    
    def test_allreduce_2_ranks(self):
        """AllReduce with 2 ranks: factor = 2*(2-1)/2 = 1.0"""
        factor = get_bus_bandwidth_factor('all_reduce', 2)
        assert factor == 1.0
    
    def test_allreduce_case_insensitive(self):
        """Test case-insensitive collective names."""
        assert get_bus_bandwidth_factor('AllReduce', 8) == 1.75
        assert get_bus_bandwidth_factor('ALLREDUCE', 8) == 1.75
        assert get_bus_bandwidth_factor('all_reduce', 8) == 1.75
    
    def test_reduce_scatter(self):
        """ReduceScatter: factor = (N-1)/N"""
        factor = get_bus_bandwidth_factor('reduce_scatter', 8)
        assert factor == 0.875  # 7/8
    
    def test_all_gather(self):
        """AllGather: factor = (N-1)/N"""
        factor = get_bus_bandwidth_factor('all_gather', 8)
        assert factor == 0.875  # 7/8
    
    def test_reduce_no_amplification(self):
        """Reduce: factor = 1.0 (no amplification)"""
        factor = get_bus_bandwidth_factor('reduce', 8)
        assert factor == 1.0
    
    def test_broadcast_no_amplification(self):
        """Broadcast: factor = 1.0 (no amplification)"""
        factor = get_bus_bandwidth_factor('broadcast', 8)
        assert factor == 1.0
    
    def test_alltoall(self):
        """AlltoAll: factor = (N-1)/N"""
        factor = get_bus_bandwidth_factor('alltoall', 8)
        assert factor == 0.875
    
    def test_unknown_collective(self):
        """Unknown collective defaults to factor = 1.0"""
        factor = get_bus_bandwidth_factor('unknown_op', 8)
        assert factor == 1.0


class TestBusBandwidth:
    """Tests for complete bus bandwidth calculation."""
    
    def test_allreduce_calculation(self):
        """Test bus BW = alg BW * factor for AllReduce."""
        # 1000 bytes in 50 µs, 8 ranks
        # algbw = 0.02 GB/s, factor = 1.75
        # busbw = 0.035 GB/s
        result = calculate_bus_bandwidth(1000, 50, 'all_reduce', 8)
        assert result == 0.035
    
    def test_reduce_scatter_calculation(self):
        """Test bus BW for ReduceScatter."""
        # algbw = 0.02 GB/s, factor = 0.875
        # busbw = 0.0175 GB/s
        result = calculate_bus_bandwidth(1000, 50, 'reduce_scatter', 8)
        assert result == 0.0175
    
    def test_reduce_calculation(self):
        """Test bus BW for Reduce (no amplification)."""
        # algbw = busbw = 0.02 GB/s
        result = calculate_bus_bandwidth(1000, 50, 'reduce', 8)
        assert result == 0.02
    
    def test_vectorized_bus_bandwidth(self):
        """Test bus BW with arrays."""
        sizes = np.array([1000, 2000, 3000])
        times = np.array([50, 100, 150])
        result = calculate_bus_bandwidth(sizes, times, 'all_reduce', 8)
        expected = np.array([0.035, 0.035, 0.035])
        np.testing.assert_array_equal(result, expected)
    
    def test_different_rank_counts(self):
        """Verify factor changes with rank count."""
        size, time = 1000, 50  # algbw = 0.02
        
        # 2 ranks: factor = 1.0, busbw = 0.02
        bw2 = calculate_bus_bandwidth(size, time, 'all_reduce', 2)
        assert bw2 == 0.02
        
        # 4 ranks: factor = 1.5, busbw = 0.03
        bw4 = calculate_bus_bandwidth(size, time, 'all_reduce', 4)
        assert bw4 == 0.03
        
        # 8 ranks: factor = 1.75, busbw = 0.035
        bw8 = calculate_bus_bandwidth(size, time, 'all_reduce', 8)
        assert bw8 == 0.035


class TestNumericalStability:
    """Tests for numerical edge cases and stability."""
    
    def test_very_small_values(self):
        """Test with very small sizes and times."""
        result = calculate_algorithm_bandwidth(1, 1)
        assert result == 0.001
    
    def test_very_large_values(self):
        """Test with very large sizes and times."""
        result = calculate_algorithm_bandwidth(1e12, 1e6)
        assert result == 1000.0  # 1e12 / 1e6 / 1000
    
    def test_floating_point_precision(self):
        """Verify floating point calculations are stable."""
        # Test case that might expose rounding errors
        result = calculate_algorithm_bandwidth(1024*1024, 33.33)
        # 1048576 / 33.33 / 1000 = 31.46
        assert abs(result - 31.46) < 0.1
    
    def test_negative_values_not_validated(self):
        """Current implementation doesn't validate negatives."""
        # This documents current behavior - may want to add validation
        result = calculate_algorithm_bandwidth(-1000, 50)
        assert result == -0.02  # Returns negative (mathematically correct but physically invalid)


class TestRealWorldScenarios:
    """Tests based on actual benchmark scenarios."""
    
    def test_mi300x_typical_performance(self):
        """Test typical MI300X AllReduce performance."""
        # Typical: 1MB message in ~50µs with 8 GPUs
        size = 1024 * 1024  # 1 MB
        time = 50.0  # µs
        nranks = 8
        
        algbw = calculate_algorithm_bandwidth(size, time)
        busbw = calculate_bus_bandwidth(size, time, 'all_reduce', nranks)
        
        # Verify reasonable values
        assert 15 < algbw < 25  # ~20 GB/s
        assert 30 < busbw < 45  # ~35 GB/s
        assert abs(busbw / algbw - 1.75) < 0.01
    
    def test_small_message_latency_bound(self):
        """Small messages are latency-bound."""
        # 16 bytes takes ~20µs (latency dominated)
        size = 16
        time = 20.0
        
        algbw = calculate_algorithm_bandwidth(size, time)
        assert algbw < 1.0  # Very low bandwidth
    
    def test_large_message_bandwidth_bound(self):
        """Large messages are bandwidth-bound."""
        # 1GB should achieve high bandwidth
        size = 1024 * 1024 * 1024
        time = 6000.0  # 6ms for 1GB
        
        algbw = calculate_algorithm_bandwidth(size, time)
        assert algbw > 100  # Should be >100 GB/s

