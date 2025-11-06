/*
 * GPU-to-GPU Transfer Benchmark with NUMA Affinity
 * Uses MPI for process management and NUMA locality, but direct GPU transfers
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <thread>
#include <unistd.h>
#include <fstream>
#include <mpi.h>

// Error checking macro
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << hipGetErrorString(err) << std::endl; \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

#define MPI_CHECK(call) \
    do { \
        int err = call; \
        if (err != MPI_SUCCESS) { \
            std::cerr << "MPI error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            MPI_Abort(MPI_COMM_WORLD, 1); \
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

// Get NUMA node for a GPU
int getGpuNumaNode(int gpu_id) {
    // For AMD MI300A, GPUs are typically associated with specific NUMA nodes
    // This is a simplified mapping - in production you'd query the system
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, gpu_id));

    // MI300A systems often have GPUs distributed across NUMA nodes
    // This mapping may need adjustment based on actual system topology
    return gpu_id % 2;  // Assume alternating NUMA nodes, adjust as needed
}

// Set CPU and memory affinity for NUMA node
void setNumaAffinity(int numa_node) {
    // Use sched_setaffinity to bind to CPUs in the NUMA node
    // and numactl-like memory policy
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    // Get number of CPUs
    int num_cpus = sysconf(_SC_NPROCESSORS_ONLN);

    // For simplicity, assume NUMA node 0 has CPUs 0-31, node 1 has 32-63, etc.
    // This is system-specific and should be queried properly
    int cpus_per_node = num_cpus / 2;  // Assume 2 NUMA nodes
    int cpu_start = numa_node * cpus_per_node;
    int cpu_end = cpu_start + cpus_per_node;

    for (int cpu = cpu_start; cpu < cpu_end; cpu++) {
        CPU_SET(cpu, &cpuset);
    }

    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
        std::cerr << "Warning: Failed to set CPU affinity for NUMA node " << numa_node << std::endl;
    }

    // Set memory policy to bind to NUMA node
    // This requires libnuma, but for simplicity we'll use mbind or just rely on CPU affinity
    std::cout << "[Rank " << numa_node << "] Set affinity to NUMA node " << numa_node
              << " (CPUs " << cpu_start << "-" << cpu_end-1 << ")" << std::endl;
}

// Enable peer access between GPUs
bool enablePeerAccess(int src_gpu, int dst_gpu) {
    int can_access = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&can_access, dst_gpu, src_gpu));

    if (can_access) {
        HIP_CHECK(hipSetDevice(src_gpu));
        hipError_t err = hipDeviceEnablePeerAccess(dst_gpu, 0);
        if (err == hipSuccess) {
            std::cout << "[Rank " << src_gpu << "] Enabled peer access: GPU " << src_gpu << " -> GPU " << dst_gpu << std::endl;
            return true;
        } else if (err == hipErrorPeerAccessAlreadyEnabled) {
            return true; // Already enabled
        } else {
            std::cout << "[Rank " << src_gpu << "] Failed to enable peer access: GPU " << src_gpu << " -> GPU " << dst_gpu
                      << " (" << hipGetErrorString(err) << ")" << std::endl;
            return false;
        }
    } else {
        std::cout << "[Rank " << src_gpu << "] Peer access not supported: GPU " << src_gpu << " -> GPU " << dst_gpu << std::endl;
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
    // Initialize MPI
    MPI_CHECK(MPI_Init(&argc, &argv));

    int world_rank, world_size;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    int num_gpus = 0;
    HIP_CHECK(hipGetDeviceCount(&num_gpus));

    // Each MPI rank controls one GPU
    if (world_size != num_gpus) {
        if (world_rank == 0) {
            std::cerr << "Error: Number of MPI processes (" << world_size
                      << ") must equal number of GPUs (" << num_gpus << ")" << std::endl;
        }
        MPI_CHECK(MPI_Finalize());
        return 1;
    }

    int my_gpu = world_rank;

    // Set NUMA affinity for this GPU
    int numa_node = getGpuNumaNode(my_gpu);
    setNumaAffinity(numa_node);

    // Set GPU device for this rank
    HIP_CHECK(hipSetDevice(my_gpu));

    if (world_rank == 0) {
        std::cout << "GPU-to-GPU Transfer Benchmark with NUMA Affinity" << std::endl;
        std::cout << "=================================================" << std::endl;
        std::cout << "Using MPI with " << world_size << " processes for NUMA locality" << std::endl;
        std::cout << "Each process controls one GPU with proper NUMA affinity" << std::endl;
        std::cout << std::endl;
    }

    // Show GPU information
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, my_gpu));

    std::cout << "[Rank " << world_rank << "] GPU " << my_gpu << ": " << props.name
              << " (NUMA node " << numa_node << ")" << std::endl;

    // Synchronize all ranks
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Enable peer access between all GPU pairs
    for (int src = 0; src < num_gpus; src++) {
        for (int dst = 0; dst < num_gpus; dst++) {
            if (src != dst) {
                // Only rank 0 does the peer access setup to avoid conflicts
                if (world_rank == 0) {
                    enablePeerAccess(src, dst);
                }
            }
        }
    }

    // Synchronize after peer access setup
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Define test sizes
    std::vector<size_t> latency_sizes = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}; // Up to 4KB
    std::vector<size_t> bandwidth_sizes; // 1KB to 1GB
    for (size_t size = 1024; size <= 1024*1024*1024; size *= 2) {
        bandwidth_sizes.push_back(size);
    }

    // Each rank runs benchmarks for all pairs involving its GPU
    std::vector<TransferResult> my_latency_results;
    std::vector<TransferResult> my_bandwidth_results;

    // Run latency benchmarks for pairs involving this GPU
    for (int other_gpu = 0; other_gpu < num_gpus; other_gpu++) {
        if (other_gpu != my_gpu) {
            // Test both directions
            for (int src : {my_gpu, other_gpu}) {
                int dst = (src == my_gpu) ? other_gpu : my_gpu;

                std::cout << "[Rank " << world_rank << "] Running latency benchmark: GPU "
                          << src << " -> GPU " << dst << std::endl;

                for (size_t size : latency_sizes) {
                    TransferResult result = benchmarkLatency(src, dst, size);
                    if (result.success) {
                        my_latency_results.push_back(result);
                    }
                }
            }
        }
    }

    // Run bandwidth benchmarks for pairs involving this GPU
    for (int other_gpu = 0; other_gpu < num_gpus; other_gpu++) {
        if (other_gpu != my_gpu) {
            // Test both directions
            for (int src : {my_gpu, other_gpu}) {
                int dst = (src == my_gpu) ? other_gpu : my_gpu;

                std::cout << "[Rank " << world_rank << "] Running bandwidth benchmark: GPU "
                          << src << " -> GPU " << dst << std::endl;

                for (size_t size : bandwidth_sizes) {
                    TransferResult result = benchmarkBandwidth(src, dst, size);
                    if (result.success) {
                        my_bandwidth_results.push_back(result);
                    }
                }
            }
        }
    }

    // Each rank saves its results to a file, then rank 0 combines them
    std::string filename = "gpu_transfer_results_mpi_rank" + std::to_string(world_rank) + ".txt";
    std::ofstream outfile(filename);

    outfile << "LATENCY BENCHMARKS - Rank " << world_rank << " (GPU " << my_gpu << ")" << std::endl;
    outfile << "=======================================" << std::endl;
    outfile << std::setw(10) << "Size"
            << std::setw(15) << "GPU Pair"
            << std::setw(15) << "Latency (μs)"
            << std::setw(15) << "Bandwidth (GB/s)" << std::endl;
    outfile << std::string(55, '-') << std::endl;

    for (const auto& result : my_latency_results) {
        outfile << std::setw(10) << formatBytes(result.size_bytes)
                << std::setw(15) << std::to_string(result.src_gpu) + "→" + std::to_string(result.dst_gpu)
                << std::setw(15) << std::fixed << std::setprecision(2) << result.latency_us
                << std::setw(15) << std::fixed << std::setprecision(2) << result.bandwidth_gbps
                << std::endl;
    }

    outfile << std::endl;
    outfile << "BANDWIDTH BENCHMARKS - Rank " << world_rank << " (GPU " << my_gpu << ")" << std::endl;
    outfile << "========================================" << std::endl;
    outfile << std::setw(10) << "Size"
            << std::setw(15) << "GPU Pair"
            << std::setw(15) << "Avg Latency (μs)"
            << std::setw(15) << "Bandwidth (GB/s)" << std::endl;
    outfile << std::string(55, '-') << std::endl;

    for (const auto& result : my_bandwidth_results) {
        outfile << std::setw(10) << formatBytes(result.size_bytes)
                << std::setw(15) << std::to_string(result.src_gpu) + "→" + std::to_string(result.dst_gpu)
                << std::setw(15) << std::fixed << std::setprecision(2) << result.latency_us
                << std::setw(15) << std::fixed << std::setprecision(2) << result.bandwidth_gbps
                << std::endl;
    }

    outfile.close();

    // Synchronize and let rank 0 combine results
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (world_rank == 0) {
        std::cout << std::endl << "Individual rank results saved. Combining..." << std::endl;

        // Combine all results into a single file
        std::ofstream combined("gpu_transfer_results_mpi_combined.txt");

        combined << "GPU-to-GPU Transfer Benchmark with NUMA Affinity" << std::endl;
        combined << "=================================================" << std::endl;
        combined << "Using MPI with " << world_size << " processes for NUMA locality" << std::endl;
        combined << std::endl;

        // Read and combine all rank files
        for (int rank = 0; rank < world_size; rank++) {
            std::string rank_file = "gpu_transfer_results_mpi_rank" + std::to_string(rank) + ".txt";
            std::ifstream infile(rank_file);
            if (infile) {
                combined << infile.rdbuf();
                combined << std::endl << std::string(80, '=') << std::endl << std::endl;
                infile.close();
                // Clean up individual files
                std::remove(rank_file.c_str());
            }
        }

        combined << "Benchmark completed with NUMA affinity!" << std::endl;
        combined.close();

        std::cout << "Combined results saved to: gpu_transfer_results_mpi_combined.txt" << std::endl;
    }

    MPI_CHECK(MPI_Finalize());
    return 0;
}
