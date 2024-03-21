#include <iostream>
#include <cmath>
#include <vector>
#include <functional>
#include <chrono>

// Función para inicializar una matriz con valores.
std::vector<size_t> Matrix(size_t rows){
    std::vector<size_t> res(rows * rows,1);
    return res;
}

void PrintTime(std::function<void()> func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double>  diff = end - start;
    std::cout << diff.count() << " s" << std::endl;
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

    for(int size= 64; size < 2024; size *=2){
        std::vector<size_t> A = Matrix(size);
        std::vector<size_t> B = Matrix(size);
        std::vector<size_t> C = std::vector<size_t> (size * size,0);
        std::cout << "Tiempo de Multiplicacion Comun tamano " << size << ":";

        PrintTime([&]() { EfficientMatrixesMultiplier(C, A, B, size); });
        for(int Block_Size = 64; Block_Size<2024; Block_Size *= 2){    
            C = std::vector<size_t> (size * size,0);
            std::cout << "Tiempo de Multiplicacion en tamano "<< size << " Bloque de " << Block_Size << " :";
            PrintTime([&]() { blockMatrixMultiply(C, A, B, Block_Size, size); });
        }
    }
    
    return 0;
}