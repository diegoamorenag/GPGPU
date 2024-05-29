#include <iostream>
#include <cuda_runtime.h>

// kernel para transponer usando memoria compartida con columna dummy para evitar conflictos de bancos
__global__ void kernel(int *input, int *output, int width, int height) {
    __shared__ int tile[32][33]; // Agregar una columna dummy

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Cargar datos en la memoria compartida
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    __syncthreads();

    // Escribir datos transpuestos desde la memoria compartida a la global
    int transpos_x = blockIdx.y * blockDim.y + threadIdx.x;
    int transpos_y = blockIdx.x * blockDim.x + threadIdx.y;
    if (transpos_x < height && transpos_y < width) {
        output[transpos_y * height + transpos_x] = tile[threadIdx.x][threadIdx.y];
    }
}

void lanzador() {
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
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Eventos para medir el tiempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Inicia medición
    cudaEventRecord(start);

    // Lanzamiento del kernel
    kernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);

    // Finaliza medición
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calcular el tiempo transcurrido
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copia de resultados hacia el host
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    // Impresión de secciones de la matriz
    //std::cout << "Bloque de la matriz original:\n";
    //printMatrixSection(h_input, width, height, 0, 5, 0, 5);

    //std::cout << "Bloque de la matriz transpuesta:\n";
    //printMatrixSection(h_output, height, width, 0, 5, 0, 5);

    std::cout << "Tiempo de ejecucion del kernel: " << milliseconds << " ms\n";
    //std::cout << milliseconds << std::endl;

    // Liberar memoria y eventos
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    for (int i = 0; i < 10; i++) {
        lanzador();
    }
    return 0;
}
