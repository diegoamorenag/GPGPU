#include <iostream>
#include <cuda.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

// Kernel para calcular la suma exclusiva usando CUDA
__global__ void exclusive_scan(int* d_in, int* d_out, int n) {
    extern __shared__ int temp[];  // Array compartido para el almacenamiento temporal

    int thid = threadIdx.x;
    int offset = 1;

    // Copiar entrada a temp (memoria compartida)
    temp[2 * thid] = d_in[2 * thid];
    temp[2 * thid + 1] = d_in[2 * thid + 1];

    // Hacer un "build sum in place" en el árbol
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // Clear the last element
    if (thid == 0) {
        temp[n - 1] = 0;
    }

    // Traverse down the tree and build the scan
    for (int d = 1; d < n; d *= 2) {
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

    // Escribir resultado a memoria global
    d_out[2 * thid] = temp[2 * thid];
    d_out[2 * thid + 1] = temp[2 * thid + 1];
}

// Función secuencial para calcular la suma exclusiva en la CPU
void exclusive_scan_cpu(const std::vector<int>& in, std::vector<int>& out) {
    int n = in.size();
    out[0] = 0;
    for (int i = 1; i < n; ++i) {
        out[i] = out[i - 1] + in[i - 1];
    }
}

// Función para verificar errores de CUDA
void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::cerr << message << " - " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int N = 1024;  // Cambiar N según sea necesario
    int size = N * sizeof(int);

    // Generar array de longitud N con elementos consecutivos
    std::vector<int> h_in(N);
    std::iota(h_in.begin(), h_in.end(), 1);

    // Inicializar array de salida en CPU
    std::vector<int> h_out(N);
    std::vector<int> h_out_cpu(N);

    // Asignar memoria en el dispositivo
    int* d_in;
    int* d_out;
    checkCudaError(cudaMalloc((void**)&d_in, size), "Error en cudaMalloc para d_in");
    checkCudaError(cudaMalloc((void**)&d_out, size), "Error en cudaMalloc para d_out");

    // Copiar datos de entrada a memoria del dispositivo
    checkCudaError(cudaMemcpy(d_in, h_in.data(), size, cudaMemcpyHostToDevice), "Error en cudaMemcpy HostToDevice");

    // Calcular tamaño de memoria compartida
    int shared_mem_size = size;

    // Ejecutar el kernel
    exclusive_scan<<<1, N / 2, shared_mem_size>>>(d_in, d_out, N);

    // Copiar resultados de vuelta al host
    checkCudaError(cudaMemcpy(h_out.data(), d_out, size, cudaMemcpyDeviceToHost), "Error en cudaMemcpy DeviceToHost");

    // Calcular suma exclusiva en la CPU para verificar resultados
    exclusive_scan_cpu(h_in, h_out_cpu);

    // Comparar resultados
    bool success = std::equal(h_out.begin(), h_out.end(), h_out_cpu.begin());
    if (success) {
        std::cout << "Los resultados son correctos." << std::endl;
    } else {
        std::cout << "Los resultados son incorrectos." << std::endl;
    }

    // Liberar memoria del dispositivo
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}