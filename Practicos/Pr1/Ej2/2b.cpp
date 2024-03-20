#include <iostream>
#include <cmath>
#include <vector>
#include <functional>
#include <chrono>

#define A_ROWS 2048  
#define A_COLS 1024 
#define B_ROWS 1024 
#define B_COLS 512 


// Función para inicializar una matriz con valores.
std::vector<size_t> Matrix(size_t rows, size_t cols){
    std::vector<size_t> res(rows * cols,1);
    return res;
}

void PrintTime(std::function<void()> func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double>  diff = end - start;
    std::cout << diff.count() << " s" << std::endl;
}

void EfficientMatrixesMultiplier(std::vector<size_t>& C, const std::vector<size_t> A, const std::vector<size_t> B){
    for (size_t it = 0; it < A_COLS; it++) { 
        for (size_t row = 0; row < A_ROWS; row++) {
            for (size_t col = 0; col < B_COLS; col++) {
                C[row * B_COLS + col] += A[row * A_COLS + it] * B[it * B_COLS + col];
            }
        }
    }
}

void blockMatrixMultiply(
    std::vector<size_t>& C,
    const std::vector<size_t> A,
    const std::vector<size_t> B, 
    int BLOCK_SIZE) {
    for (int i = 0; i < A_ROWS; i += BLOCK_SIZE) {
        for (int j = 0; j < B_COLS; j += BLOCK_SIZE) {
            for (int k = 0; k < A_COLS; k += BLOCK_SIZE) {
                // Realiza la multiplicación de los bloques
                for (int ii = i; ii < std::min(i + BLOCK_SIZE, A_ROWS); ++ii) {
                    for (int kk = k; kk < std::min(k + BLOCK_SIZE, A_COLS); ++kk) {
                        size_t temp = A[ii * A_COLS + kk];
                        for (int jj = j; jj < std::min(j + BLOCK_SIZE, B_COLS); ++jj) {
                            C[ii * B_COLS + jj] += temp * B[kk * B_COLS + jj];
                        }
                    }
                }
            }
        }
    }
}

int main() {
    // Inicializa las matrices A, B y C.
    std::vector<size_t> A = Matrix(A_ROWS,A_COLS);
    std::vector<size_t> B = Matrix(B_ROWS,B_COLS);
    std::vector<size_t> C = std::vector<size_t> (A_ROWS * B_COLS,0);

    
    std::cout << "Tiempo de Multiplicacion Comun:";
    PrintTime([&]() { EfficientMatrixesMultiplier(C, A, B); });

    for(int i = 64; i<2024; i *= 2){
        C = std::vector<size_t> (A_ROWS * B_COLS,0);
        std::cout << "Tiempo de Multiplicacion en Bloque de " << i << " :";
        PrintTime([&]() { blockMatrixMultiply(C, A, B, i); });
    }
    
    return 0;
}