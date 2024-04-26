#include <iostream>
#include <cuda_runtime.h> // Asegúrate de incluir el header adecuado para funciones CUDA.

#define M 10240
#define N 256

// kernel para transponer
__global__ void kernel(int *A, int *v, int *x) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < M) {
        int productoInterno = 0;
        for (int i = 0; i < N; i++) {
            productoInterno += A[pos + i] * v[i];
        }
        x[pos] = productoInterno;
    }
}

// Función para imprimir una sección de la matriz
void printMatrixSection(int *matrix, int width, int height, int rowStart, int rowEnd, int colStart, int colEnd) {
    std::cout << "Matriz A: " << std::endl;
    for (int i = rowStart; i < rowEnd; i++) {
        for (int j = colStart; j < colEnd; j++) {
            std::cout << matrix[i * width + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void obtener_timepo() {
    int width = N;
    int height = M;
    size_t sizeMatrix = width * height * sizeof(int);
    size_t sizeVector = width * sizeof(int);
    size_t sizeResult = height * sizeof(int);

    // Allocate memory on the host
    int *h_A = (int*)malloc(sizeMatrix);
    int *h_v = (int*)malloc(sizeVector);
    int *h_x = (int*)malloc(sizeResult);

    // Inicialización de la matriz de entrada
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            h_A[i * width + j] = 1;
        }
    }

    printMatrixSection(h_A, width, height, 0, 5, 0, 5);

    // Inicialización del vector de entrada
    for (int i = 0; i < width; i++) {
        h_v[i] = 1;
    }

    //imprimir el vector de entrada
    std::cout << "Vector v: " << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << h_v[i] << " ";
    }
    std::cout << std::endl;

    // Allocate memory on the device
    int *d_A, *d_v, *d_x;
    cudaMalloc(&d_A, sizeMatrix);
    cudaMalloc(&d_v, sizeVector);
    cudaMalloc(&d_x, sizeResult);

    // Copy the matrix A and vector v to the device
    cudaMemcpy(d_A, h_A, sizeMatrix, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, sizeVector, cudaMemcpyHostToDevice);

    // Configuración del tamaño de bloque y de grilla
    int threadsPerBlock = 256;
    int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;

    // Crear eventos para medir el tiempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Registro del evento de inicio
    cudaEventRecord(start);

    // Lanzamiento del kernel
    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_v, d_x);

    // Registro del evento de finalización
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calcula y muestra el tiempo de ejecución
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Tiempo de ejecución: " << milliseconds << " ms\n";

    // Copy the result vector x back to the host
    cudaMemcpy(h_x, d_x, sizeResult, cudaMemcpyDeviceToHost);

    // Print the result vector x
    std::cout << "Vector x: " << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << h_x[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_v);
    cudaFree(d_x);

    // Free host memory
    free(h_A);
    free(h_v);
    free(h_x);

    // Destruir eventos
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return;
}


int main (void) {
    obtener_timepo();
    return 0;
}