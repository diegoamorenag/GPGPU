#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>

// Kernel para transponer
__global__ void kernel(int *input, int *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int pos = y * width + x;
        int transpos = x * height + y;
        output[transpos] = input[pos];
    }
}

// Función para obtener tiempo promedio y desviación estándar
void obtener_tiempo(int block_x, int block_y) {
    const int num_runs = 10;
    std::vector<float> runtimes(num_runs);
    int width = 16384;
    int height = 16384;
    size_t bytes = width * height * sizeof(int);

    int *h_input, *h_output;
    int *d_input, *d_output;

    // Reserva de memoria en el host
    h_input = (int*)malloc(bytes);
    h_output = (int*)malloc(bytes);

    // Inicialización de la matriz de entrada
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            h_input[i * width + j] = i * width + j;
        }
    }

    // Reserva de memoria en el device
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    // Copia de datos hacia el device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // Configuración del tamaño de bloque y de grilla
    dim3 blockSize(block_x, block_y);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Crear eventos para medir el tiempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int run = 0; run < num_runs; run++) {
        // Registro del evento de inicio
        cudaEventRecord(start);

        // Lanzamiento del kernel
        kernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);

        // Registro del evento de finalización
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Calcula y guarda el tiempo de ejecución
        cudaEventElapsedTime(&runtimes[run], start, stop);
    }

    // Copia de resultados hacia el host
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    // Liberar memoria
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    // Destruir eventos
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Calcular media y desviación estándar
    float mean = 0;
    for (float time : runtimes) {
        mean += time;
    }
    mean /= num_runs;

    float stddev = 0;
    for (float time : runtimes) {
        stddev += (time - mean) * (time - mean);
    }
    stddev = sqrt(stddev / num_runs);

    //std::cout << "Tiempo promedio de ejecucion del kernel para (" << block_x << ", " << block_y << ") es: " << mean << " ms, con una desviación estándar de: " << stddev << " ms\n";
    std::cout << "(" << block_x << "," << block_y << ") " << mean << " " << stddev << "\n";
}

int main(void) {
    for (int i = 1; i <= 256; i *= 2) {
        for (int j = 1; j <= 256; j *= 2) {
            obtener_tiempo(i, j);
        }
    }
    return 0;
}
