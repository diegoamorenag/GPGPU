#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

#include "../auxFunctions.h"

// Función para inicializar una matriz con valores.
std::vector<size_t> Matrix(size_t rows){
    std::vector<size_t> res(rows * rows,1);
    return res;
}

void EfficientMatrixesMultiplier(std::vector<size_t>& C, const std::vector<size_t> A, const std::vector<size_t> B, int size){
    for (size_t it = 0; it < size; it++) { 
        for (size_t row = 0; row < size; row++) {
            for (size_t col = 0; col < size; col++) {
                C[row * size + col] += A[row * size + it] * B[it * size + col];
            }
        }
    }
}

void blockMatrixMultiply(
    std::vector<size_t>& C,
    const std::vector<size_t> A,
    const std::vector<size_t> B, 
    int BLOCK_SIZE,
    int size) {
    for (int i = 0; i < size; i += BLOCK_SIZE) {
        for (int j = 0; j < size; j += BLOCK_SIZE) {
            for (int k = 0; k < size; k += BLOCK_SIZE) {
                // Realiza la multiplicación de los bloques
                for (int ii = i; ii < std::min(i + BLOCK_SIZE, size); ++ii) {
                    for (int kk = k; kk < std::min(k + BLOCK_SIZE, size); ++kk) {
                        size_t temp = A[ii * size + kk];
                        for (int jj = j; jj < std::min(j + size, size); ++jj) {
                            C[ii * size + jj] += temp * B[kk * size + jj];
                        }
                    }
                }
            }
        }
    }
}

int main() {

    system("mkdir -p Ej2/results");

    std::ofstream results("Ej2/results/2c");

    int conflictBlockSize = 512; // Ejemplo para conflictos con L1 caché

    for (int size = 512; size <= 2048; size *= 2) {
        // ... (inicialización de matrices A, B, C)
        std::vector<size_t> A = Matrix(size);
        std::vector<size_t> B = Matrix(size);
        std::vector<size_t> C = std::vector<size_t> (size * size,0);

        // Medir el rendimiento con un tamaño de bloque que causa conflictos
        auto start = std::chrono::high_resolution_clock::now();
        blockMatrixMultiply(C, A, B, conflictBlockSize, size);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        results << "Tiempo con conflictos de caché tamaño " << size << " y bloque " << conflictBlockSize << ": " << diff.count() << " s\n";
    }
    results.close();
    return 0;
}
