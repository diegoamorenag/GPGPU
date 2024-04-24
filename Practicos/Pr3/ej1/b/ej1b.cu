#include <iostream>
#include <cuda_runtime.h> // Asegúrate de incluir el header adecuado para funciones CUDA.

// kernel para transponer
__global__ void kernel(int *input, int *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int pos = y * width + x;
        int transpos = x * height + y;
        output[transpos] = input[pos];
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

int main() {
    int width = 1024; // Asumiendo un tamaño de matriz de 1024x1024
    int height = 1024;
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
    dim3 blockSize(32, 8);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Crear eventos para medir el tiempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Registro del evento de inicio
    cudaEventRecord(start);

    // Lanzamiento del kernel
    kernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);

    // Registro del evento de finalización
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calcula y muestra el tiempo de ejecución
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Tiempo de ejecución del kernel: " << milliseconds << " ms\n";

    // Copia de resultados hacia el host
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    // Impresión de secciones de la matriz
    std::cout << "Bloque de la matriz original:\n";
    printMatrixSection(h_input, width, height, 0, 5, 0, 5);
    std::cout << "Bloque de la matriz transpuesta:\n";
    printMatrixSection(h_output, width, height, 0, 5, 0, 5);

    // Liberar memoria
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    // Destruir eventos
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
