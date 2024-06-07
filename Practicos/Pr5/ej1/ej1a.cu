#include <iostream>
#include <cuda.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>


__global__ void exclusive_scan_kernel(float* input, float* output, int n) {
    extern __shared__ float temp[];  // Allocated memory for shared data
    int thid = threadIdx.x;
    int offset = 1;

    // Load input into shared memory.
    // Each thread loads one element
    int ai = thid;
    int bi = thid + (n/2);
    temp[ai] = (ai < n) ? input[ai] : 0;
    temp[bi] = (bi < n) ? input[bi] : 0;

    __syncthreads();

    // Build sum in place up the tree
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // Clear the last element
    if (thid == 0) { temp[n - 1] = 0; }

    // Traverse down tree & build scan
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // Write results to output array
    if (ai < n) output[ai] = temp[ai];
    if (bi < n) output[bi] = temp[bi];
}

// Función secuencial para calcular la suma exclusiva en la CPU
void exclusive_scan_cpu(const std::vector<int>& in, std::vector<int>& out) {
    out[0] = 0;
    for (size_t i = 1; i < in.size(); ++i) {
        out[i] = out[i - 1] + in[i - 1];
    }
}

// Function to check CUDA errors
void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

// Calcula la desviación estándar de los tiempos
double calculate_stddev(const std::vector<float>& times, float mean) {
    double sum = 0.0;
    for (auto t : times) {
        sum += (t - mean) * (t - mean);
    }
    return std::sqrt(sum / times.size());
}

int main() {
    for (int exp = 0; exp <= 10; ++exp) {
        int N = 1024;
        int size = N * pow(2, exp);
        std::vector<int> h_in(size), h_out(size), h_out_cpu(size);
        std::iota(h_in.begin(), h_in.end(), 0);  // Genera valores consecutivos desde 0

        int* d_in;
        int* d_out;
        int* d_sums;
        int num_blocks = (size + N - 1) / N;
        
        checkCudaError(cudaMalloc((void**)&d_in, size * sizeof(int)), "Error en cudaMalloc para d_in");
        checkCudaError(cudaMalloc((void**)&d_out, size * sizeof(int)), "Error en cudaMalloc para d_out");
        checkCudaError(cudaMalloc((void**)&d_sums, num_blocks * sizeof(int)), "Error en cudaMalloc para d_sums");

        checkCudaError(cudaMemcpy(d_in, h_in.data(), size * sizeof(int), cudaMemcpyHostToDevice), "Error en cudaMemcpy HostToDevice");

        int shared_mem_size = N * sizeof(int);
        std::vector<float> times(10);

        cudaEvent_t start, stop;
        checkCudaError(cudaEventCreate(&start), "Error creating start event");
        checkCudaError(cudaEventCreate(&stop), "Error creating stop event");

        int threads = N / 2;  // Number of threads should match the base size / 2

        for (int i = 0; i < 10; ++i) {
            cudaEventRecord(start);
            exclusive_scan_kernel<<<num_blocks, threads, shared_mem_size>>>(d_in, d_out, size, d_sums);
            cudaEventSynchronize(start);
            adjust_sums<<<num_blocks, threads>>>(d_out, d_sums, size);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            times[i] = milliseconds;
        }

        float mean_time = std::accumulate(times.begin(), times.end(), 0.0f) / times.size();
        double std_dev = calculate_stddev(times, mean_time);

        checkCudaError(cudaMemcpy(h_out.data(), d_out, size * sizeof(int), cudaMemcpyDeviceToHost), "Error en cudaMemcpy DeviceToHost");

        // Verificar con la implementación secuencial
        exclusive_scan_cpu(h_in, h_out_cpu);
        bool correct = std::equal(h_out.begin(), h_out.end(), h_out_cpu.begin());
        if (!correct) {
            std::cerr << "Resultado incorrecto para tamaño " << size << std::endl;
        }

        std::cout << "Size: " << N << "*2^" << exp << ", Media: " << mean_time * 1000 << " microsec, Desvio: " << std_dev * 1000 << " microsec" << std::endl;

        cudaFree(d_in);
        cudaFree(d_out);
        cudaFree(d_sums);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    return 0;
}
