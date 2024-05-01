#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#define M 10240
#define N 256

__global__ void kernel(int *A, int *v, int *x) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < M) {
        int productoInterno = 0;
        for (int i = 0; i < N; i++) {
            productoInterno += A[pos * N + i] * v[i];
        }
        x[pos] = productoInterno;
    }
}

void ejecutar_experimentos() {
    int width = N;
    int height = M;
    size_t sizeMatrix = width * height * sizeof(int);
    size_t sizeVector = width * sizeof(int);
    size_t sizeResult = height * sizeof(int);

    int *h_A = (int*)malloc(sizeMatrix);
    int *h_v = (int*)malloc(sizeVector);
    int *h_x = (int*)malloc(sizeResult);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            h_A[i * width + j] = 1;
        }
    }
    for (int i = 0; i < width; i++) {
        h_v[i] = 1;
    }

    int *d_A, *d_v, *d_x;
    cudaMalloc(&d_A, sizeMatrix);
    cudaMalloc(&d_v, sizeVector);
    cudaMalloc(&d_x, sizeResult);

    cudaMemcpy(d_A, h_A, sizeMatrix, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, sizeVector, cudaMemcpyHostToDevice);

    for (int threadsPerBlock = 1; threadsPerBlock <= 1; threadsPerBlock *= 2) {
        int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;
        float times[100];
        for (int n = 0; n < 100; n++) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_v, d_x);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&times[n], start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        float sum = 0.0, mean, stdDev = 0.0;
        for (int i = 0; i < 10; i++) {
            sum += times[i];
        }
        mean = sum / 10;

        for (int i = 0; i < 10; i++) {
            stdDev += pow(times[i] - mean, 2);
        }
        stdDev = sqrt(stdDev / 10);

        //std::cout << "Threads per Block: " << threadsPerBlock << ", Mean: " << mean << " ms, Std Dev: " << stdDev << " ms\n";
        std::cout << threadsPerBlock << " " << mean << " " << stdDev << "\n";
    }

    cudaFree(d_A);
    cudaFree(d_v);
    cudaFree(d_x);
    free(h_A);
    free(h_v);
    free(h_x);
}

int main() {
    ejecutar_experimentos();
    return 0;
}