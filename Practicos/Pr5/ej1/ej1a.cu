#include <iostream>
#include <vector>
#include <cuda.h>
#include <algorithm>

// CUDA Kernel to perform block-wise exclusive scan
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

// CUDA Kernel to add block sums to each element
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

// Function to perform exclusive scan using CUDA
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
    }

    cudaMemcpy(output.data(), d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_block_sums);
}

// Function to perform exclusive scan sequentially
void exclusive_scan_sequential(const std::vector<int>& input, std::vector<int>& output) {
    int n = input.size();
    output[0] = 0;
    for (int i = 1; i < n; ++i) {
        output[i] = output[i - 1] + input[i - 1];
    }
}

int main() {
    std::vector<int> Ns = {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288};
    
    for (int n : Ns) {
        std::vector<int> input(n);
        std::vector<int> output_gpu(n);
        std::vector<int> output_cpu(n);

        for (int i = 0; i < n; ++i) {
            input[i] = i + 1;
        }

        exclusive_scan(input, output_gpu);
        exclusive_scan_sequential(input, output_cpu);

        bool are_equal = std::equal(output_gpu.begin(), output_gpu.end(), output_cpu.begin());
        std::cout << "N = " << n << " -> " << (are_equal ? "Iguales" : "Diferentes") << std::endl;
    }

    return 0;
}
