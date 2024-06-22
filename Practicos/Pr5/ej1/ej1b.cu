#include <iostream>
#include <vector>
#include <cuda.h>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <cmath>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>

// Custom exclusive scan implementation
__global__ void block_scan_kernel(int* d_input, int* d_output, int* d_block_sums, int n) {
    extern __shared__ int temp[];
    int thid = threadIdx.x;
    int offset = 1;

    int block_start = 2 * blockIdx.x * blockDim.x;
    if (block_start + 2 * thid < n) {
        temp[2 * thid] = d_input[block_start + 2 * thid];
    } else {
        temp[2 * thid] = 0;
    }
    if (block_start + 2 * thid + 1 < n) {
        temp[2 * thid + 1] = d_input[block_start + 2 * thid + 1];
    } else {
        temp[2 * thid + 1] = 0;
    }
    __syncthreads();

    for (int d = blockDim.x; d > 0; d >>= 1) {
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
        __syncthreads();
    }

    if (thid == 0) {
        d_block_sums[blockIdx.x] = temp[2 * blockDim.x - 1];
        temp[2 * blockDim.x - 1] = 0;
    }

    for (int d = 1; d < 2 * blockDim.x; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    if (block_start + 2 * thid < n) {
        d_output[block_start + 2 * thid] = temp[2 * thid];
    }
    if (block_start + 2 * thid + 1 < n) {
        d_output[block_start + 2 * thid + 1] = temp[2 * thid + 1];
    }
}

__global__ void add_block_sums_kernel(int* d_output, int* d_block_sums, int n) {
    int thid = threadIdx.x;
    int block_start = 2 * blockIdx.x * blockDim.x;
    if (blockIdx.x > 0) {
        if (block_start + 2 * thid < n) {
            d_output[block_start + 2 * thid] += d_block_sums[blockIdx.x];
        }
        if (block_start + 2 * thid + 1 < n) {
            d_output[block_start + 2 * thid + 1] += d_block_sums[blockIdx.x];
        }
    }
}

void exclusive_scan(const std::vector<int>& input, std::vector<int>& output) {
    int n = input.size();
    int* d_input = nullptr;
    int* d_output = nullptr;
    int* d_block_sums = nullptr;

    int blockSize = 512;  // Using 512 threads per block for better performance
    int numBlocks = (n + 2 * blockSize - 1) / (2 * blockSize);

    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));
    cudaMalloc(&d_block_sums, numBlocks * sizeof(int));
    cudaMemcpy(d_input, input.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    int sharedMemorySize = 2 * blockSize * sizeof(int);

    block_scan_kernel<<<numBlocks, blockSize, sharedMemorySize>>>(d_input, d_output, d_block_sums, n);
    cudaDeviceSynchronize();

    if (numBlocks > 1) {
        std::vector<int> block_sums(numBlocks);
        std::vector<int> block_sums_scan(numBlocks);
        cudaMemcpy(block_sums.data(), d_block_sums, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

        block_sums_scan[0] = 0;
        for (int i = 1; i < numBlocks; ++i) {
            block_sums_scan[i] = block_sums_scan[i - 1] + block_sums[i - 1];
        }

        cudaMemcpy(d_block_sums, block_sums_scan.data(), numBlocks * sizeof(int), cudaMemcpyHostToDevice);
        add_block_sums_kernel<<<numBlocks, blockSize>>>(d_output, d_block_sums, n);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(output.data(), d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_block_sums);
}

void exclusive_scan_sequential(const std::vector<int>& input, std::vector<int>& output) {
    int n = input.size();
    output[0] = 0;
    for (int i = 1; i < n; ++i) {
        output[i] = output[i - 1] + input[i - 1];
    }
}

// CUB implementation of exclusive scan
void exclusive_scan_cub(const std::vector<int>& input, std::vector<int>& output) {
    int n = input.size();
    int* d_input = nullptr;
    int* d_output = nullptr;

    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));
    cudaMemcpy(d_input, input.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_input, d_output, n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_input, d_output, n);

    cudaMemcpy(output.data(), d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp_storage);
}

// Thrust implementation of exclusive scan
void exclusive_scan_thrust(const std::vector<int>& input, std::vector<int>& output) {
    thrust::host_vector<int> h_input(input.begin(), input.end());
    thrust::device_vector<int> d_input = h_input;
    thrust::device_vector<int> d_output(input.size());

    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin());

    thrust::copy(d_output.begin(), d_output.end(), output.begin());
}

void calc_mean_stddev(const std::vector<double>& times, double& mean, double& stddev) {
    mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
    stddev = std::sqrt(sq_sum / times.size() - mean * mean);
}

int main() {
    std::vector<int> exponents = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};  // Exponents for 2^N

    // Warm-up kernel execution
    std::vector<int> warmup_input(1024, 1);
    std::vector<int> warmup_output(1024);
    exclusive_scan(warmup_input, warmup_output);

    std::cout << "N\tmedia_custom\tmedia_CUB\tmedia_thrust\n";

    for (int exp : exponents) {
        int n = 1024 * (1 << exp);  // Calculate 1024 * 2^N
        std::vector<int> input(n);
        std::vector<int> output_gpu(n);
        std::vector<int> output_cpu(n);
        std::vector<int> output_cub(n);
        std::vector<int> output_thrust(n);

        for (int i = 0; i < n; ++i) {
            input[i] = i + 1;
        }

        std::vector<double> times_custom;
        for (int i = 0; i < 10; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            exclusive_scan(input, output_gpu);
            cudaDeviceSynchronize();  // Ensure kernel execution is complete
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            times_custom.push_back(duration.count());
        }

        std::vector<double> times_cub;
        for (int i = 0; i < 10; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            exclusive_scan_cub(input, output_cub);
            cudaDeviceSynchronize();  // Ensure kernel execution is complete
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            times_cub.push_back(duration.count());
        }

        std::vector<double> times_thrust;
        for (int i = 0; i < 10; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            exclusive_scan_thrust(input, output_thrust);
            cudaDeviceSynchronize();  // Ensure kernel execution is complete
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            times_thrust.push_back(duration.count());
        }

        double mean_custom, stddev_custom;
        calc_mean_stddev(times_custom, mean_custom, stddev_custom);

        double mean_cub, stddev_cub;
        calc_mean_stddev(times_cub, mean_cub, stddev_cub);

        double mean_thrust, stddev_thrust;
        calc_mean_stddev(times_thrust, mean_thrust, stddev_thrust);

        std::cout << exp << "\t" << mean_custom << "\t" << mean_cub << "\t" << mean_thrust << "\n";
    }

    return 0;
}
