#include <iostream>
#include <cuda_runtime.h> // Asegúrate de incluir el header adecuado para funciones CUDA.

__global__ void kernel(int *matrix, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x + 4 < width && y < height) {
        int currentIdx = y * width + x;
        int targetIdx = y * width + (x + 4);
        matrix[currentIdx] += matrix[targetIdx];
    }
}

// Función para imprimir una sección de la matriz
void printMatrixSection(int *matrix, int width, int height, int rowStart, int rowEnd, int colStart, int colEnd) {
    for (int i = rowStart; i < rowEnd; i++) {
        for (int j = colStart; j < colEnd; j++) {
            std::cout << matrix[i * width + j] << " ";
        }
        std::cout << std::endl;
    }
}

void obtener_timepo(int block_x, int block_y) {
    int width = 4096; // Asumiendo un tamaño de matriz de 1024x1024
    int height = 4096;
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

    // Registro del evento de inicio
    cudaEventRecord(start);

    // Lanzamiento del kernel
    kernel<<<gridSize, blockSize>>>(d_input, width, height);

    // Registro del evento de finalización
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calcula y muestra el tiempo de ejecución
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Tiempo de ejecución del kernel para (" << block_x << ", " << block_y << ") tomo: " << milliseconds << " ms\n";

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

    return;
}


int main (void) {
    for (int i = 1; i <= 32; i*=2){
        for (int j = 1; j <= 32; j*=2){
            obtener_timepo(i, j);
        }
    }
    return 0;
}