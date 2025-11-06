/*
 * GPU-to-GPU Transfer Benchmark
 * Measures latency and bandwidth of GPU interconnect independently of RCCL
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>

// Error checking macro
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << hipGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while (0)

struct TransferResult {
    size_t size_bytes;
    double latency_us;
    double bandwidth_gbps;
    int src_gpu;
    int dst_gpu;
    bool success;
};

// Enable peer access between GPUs
bool enablePeerAccess(int src_gpu, int dst_gpu) {
    int can_access = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&can_access, dst_gpu, src_gpu));

    if (can_access) {
        HIP_CHECK(hipSetDevice(src_gpu));
        hipError_t err = hipDeviceEnablePeerAccess(dst_gpu, 0);
        if (err == hipSuccess) {
            std::cout << "Enabled peer access: GPU " << src_gpu << " -> GPU " << dst_gpu << std::endl;
            return true;
        } else if (err == hipErrorPeerAccessAlreadyEnabled) {
            return true; // Already enabled
        } else {
            std::cout << "Failed to enable peer access: GPU " << src_gpu << " -> GPU " << dst_gpu
                      << " (" << hipGetErrorString(err) << ")" << std::endl;
            return false;
        }
    } else {
        std::cout << "Peer access not supported: GPU " << src_gpu << " -> GPU " << dst_gpu << std::endl;
        return false;
    }
}

// Benchmark latency (small transfers)
TransferResult benchmarkLatency(int src_gpu, int dst_gpu, size_t size_bytes, int iterations = 1000) {
    TransferResult result = {size_bytes, 0.0, 0.0, src_gpu, dst_gpu, false};

    // Allocate memory on both GPUs
    void *src_ptr = nullptr, *dst_ptr = nullptr;

    HIP_CHECK(hipSetDevice(src_gpu));
    HIP_CHECK(hipMalloc(&src_ptr, size_bytes));

    HIP_CHECK(hipSetDevice(dst_gpu));
    HIP_CHECK(hipMalloc(&dst_ptr, size_bytes));

    // Initialize source data
    HIP_CHECK(hipSetDevice(src_gpu));
    HIP_CHECK(hipMemset(src_ptr, 0xAB, size_bytes));

    // Create events for timing
    hipEvent_t start_event, stop_event;
    HIP_CHECK(hipEventCreate(&start_event));
    HIP_CHECK(hipEventCreate(&stop_event));

    // Warmup
    for (int i = 0; i < 10; i++) {
        HIP_CHECK(hipMemcpy(dst_ptr, src_ptr, size_bytes, hipMemcpyDeviceToDevice));
    }

    // Timed iterations
    HIP_CHECK(hipEventRecord(start_event));
    for (int i = 0; i < iterations; i++) {
        HIP_CHECK(hipMemcpy(dst_ptr, src_ptr, size_bytes, hipMemcpyDeviceToDevice));
    }
    HIP_CHECK(hipEventRecord(stop_event));
    HIP_CHECK(hipEventSynchronize(stop_event));

    float elapsed_ms;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start_event, stop_event));

    result.latency_us = (elapsed_ms * 1000.0) / iterations;
    result.bandwidth_gbps = (size_bytes * iterations) / (elapsed_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0) * 8.0;
    result.success = true;

    // Cleanup
    HIP_CHECK(hipFree(src_ptr));
    HIP_CHECK(hipFree(dst_ptr));
    HIP_CHECK(hipEventDestroy(start_event));
    HIP_CHECK(hipEventDestroy(stop_event));

    return result;
}

// Benchmark bandwidth (large transfers)
TransferResult benchmarkBandwidth(int src_gpu, int dst_gpu, size_t size_bytes, int iterations = 10) {
    TransferResult result = {size_bytes, 0.0, 0.0, src_gpu, dst_gpu, false};

    // Allocate memory on both GPUs
    void *src_ptr = nullptr, *dst_ptr = nullptr;

    HIP_CHECK(hipSetDevice(src_gpu));
    HIP_CHECK(hipMalloc(&src_ptr, size_bytes));

    HIP_CHECK(hipSetDevice(dst_gpu));
    HIP_CHECK(hipMalloc(&dst_ptr, size_bytes));

    // Initialize source data
    HIP_CHECK(hipSetDevice(src_gpu));
    HIP_CHECK(hipMemset(src_ptr, 0xAB, size_bytes));

    // Create events for timing
    hipEvent_t start_event, stop_event;
    HIP_CHECK(hipEventCreate(&start_event));
    HIP_CHECK(hipEventCreate(&stop_event));

    // Warmup
    for (int i = 0; i < 3; i++) {
        HIP_CHECK(hipMemcpy(dst_ptr, src_ptr, size_bytes, hipMemcpyDeviceToDevice));
    }

    // Timed iterations
    HIP_CHECK(hipEventRecord(start_event));
    for (int i = 0; i < iterations; i++) {
        HIP_CHECK(hipMemcpy(dst_ptr, src_ptr, size_bytes, hipMemcpyDeviceToDevice));
    }
    HIP_CHECK(hipEventRecord(stop_event));
    HIP_CHECK(hipEventSynchronize(stop_event));

    float elapsed_ms;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start_event, stop_event));

    double total_bytes = static_cast<double>(size_bytes) * iterations;
    double total_time_sec = elapsed_ms / 1000.0;

    result.latency_us = elapsed_ms * 1000.0 / iterations; // Average latency per transfer
    result.bandwidth_gbps = total_bytes / total_time_sec / (1024.0 * 1024.0 * 1024.0) * 8.0;
    result.success = true;

    // Cleanup
    HIP_CHECK(hipFree(src_ptr));
    HIP_CHECK(hipFree(dst_ptr));
    HIP_CHECK(hipEventDestroy(start_event));
    HIP_CHECK(hipEventDestroy(stop_event));

    return result;
}

std::string formatBytes(size_t bytes) {
    if (bytes >= 1024 * 1024 * 1024) {
        return std::to_string(bytes / (1024 * 1024 * 1024)) + "GB";
    } else if (bytes >= 1024 * 1024) {
        return std::to_string(bytes / (1024 * 1024)) + "MB";
    } else if (bytes >= 1024) {
        return std::to_string(bytes / 1024) + "KB";
    } else {
        return std::to_string(bytes) + "B";
    }
}

int main(int argc, char* argv[]) {
    int num_gpus = 0;
    HIP_CHECK(hipGetDeviceCount(&num_gpus));

    std::cout << "GPU-to-GPU Transfer Benchmark" << std::endl;
    std::cout << "==============================" << std::endl;
    std::cout << "Found " << num_gpus << " GPUs" << std::endl;
    std::cout << std::endl;

    // Show GPU information
    for (int i = 0; i < num_gpus; i++) {
        hipDeviceProp_t props;
        HIP_CHECK(hipGetDeviceProperties(&props, i));
        std::cout << "GPU " << i << ": " << props.name << std::endl;
    }
    std::cout << std::endl;

    // Test peer access between all GPU pairs
    std::vector<std::pair<int, int>> gpu_pairs;
    for (int src = 0; src < num_gpus; src++) {
        for (int dst = 0; dst < num_gpus; dst++) {
            if (src != dst && enablePeerAccess(src, dst)) {
                gpu_pairs.emplace_back(src, dst);
            }
        }
    }

    if (gpu_pairs.empty()) {
        std::cout << "No GPU pairs support peer access. Exiting." << std::endl;
        return 1;
    }

    std::cout << "Testing " << gpu_pairs.size() << " GPU pairs" << std::endl;
    std::cout << std::endl;

    // Define test sizes
    std::vector<size_t> latency_sizes = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}; // Up to 4KB
    std::vector<size_t> bandwidth_sizes; // 1KB to 1GB
    for (size_t size = 1024; size <= 1024*1024*1024; size *= 2) {
        bandwidth_sizes.push_back(size);
    }

    // Run latency benchmarks
    std::cout << "LATENCY BENCHMARKS" << std::endl;
    std::cout << "==================" << std::endl;
    std::cout << std::setw(10) << "Size"
              << std::setw(15) << "GPU Pair"
              << std::setw(15) << "Latency (μs)"
              << std::setw(15) << "Bandwidth (GB/s)" << std::endl;
    std::cout << std::string(55, '-') << std::endl;

    for (auto& pair : gpu_pairs) {
        for (size_t size : latency_sizes) {
            TransferResult result = benchmarkLatency(pair.first, pair.second, size);
            if (result.success) {
                std::cout << std::setw(10) << formatBytes(size)
                          << std::setw(15) << std::to_string(pair.first) + "→" + std::to_string(pair.second)
                          << std::setw(15) << std::fixed << std::setprecision(2) << result.latency_us
                          << std::setw(15) << std::fixed << std::setprecision(2) << result.bandwidth_gbps
                          << std::endl;
            }
        }
        std::cout << std::endl;
    }

    // Run bandwidth benchmarks
    std::cout << "BANDWIDTH BENCHMARKS" << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << std::setw(10) << "Size"
              << std::setw(15) << "GPU Pair"
              << std::setw(15) << "Avg Latency (μs)"
              << std::setw(15) << "Bandwidth (GB/s)" << std::endl;
    std::cout << std::string(55, '-') << std::endl;

    for (auto& pair : gpu_pairs) {
        for (size_t size : bandwidth_sizes) {
            TransferResult result = benchmarkBandwidth(pair.first, pair.second, size);
            if (result.success) {
                std::cout << std::setw(10) << formatBytes(size)
                          << std::setw(15) << std::to_string(pair.first) + "→" + std::to_string(pair.second)
                          << std::setw(15) << std::fixed << std::setprecision(2) << result.latency_us
                          << std::setw(15) << std::fixed << std::setprecision(2) << result.bandwidth_gbps
                          << std::endl;
            }
        }
        std::cout << std::endl;
    }

    std::cout << "Benchmark completed!" << std::endl;
    return 0;
}
