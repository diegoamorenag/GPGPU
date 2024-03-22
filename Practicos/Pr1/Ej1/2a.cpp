#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include "../auxFunctions.h"



#define L3_SIZE (16 * 1024 * 1024) // Tamaño de la caché L3 en bytes
#define A_ROWS 2048  
#define A_COLS 1024 
#define B_ROWS 1024 
#define B_COLS 512 

std::vector<size_t> Matrix(size_t rows, size_t cols){
    std::vector<size_t> res(rows * cols,1);
    return res;
}

void EfficientMatrixesMultiplier(std::vector<size_t>& C, const std::vector<size_t>& A, const std::vector<size_t>& B){
    for (size_t it = 0; it < A_COLS; it++) { 
        for (size_t row = 0; row < A_ROWS; row++) {
            for (size_t col = 0; col < B_COLS; col++) {
                C[row * B_COLS + col] += A[row * A_COLS + it] * B[it * B_COLS + col];
            }
        }
    }
}
void NonEfficientMatrixesMultiplier(std::vector<size_t>& C, const std::vector<size_t>& A, const std::vector<size_t>& B){
    for (size_t row = 0; row < A_ROWS; row++) {
        for (size_t col = 0; col < B_COLS; col++) {
            for (size_t it = 0; it < A_COLS; it++) { 
                C[row * B_COLS + col] += A[row * A_COLS + it] * B[it * B_COLS + col];
            }
        }
    }
}

void printC(const std::vector<size_t> matrix){
    for (int i = 0; i < A_ROWS; ++i) {
        for (int j = 0; j < B_COLS; ++j) {  
            std::cout << matrix[i * A_COLS + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    system("mkdir -p Ej2/results");

    std::ofstream results("Ej2/results/2a");
    
    std::vector<size_t> A = Matrix(A_ROWS,A_COLS);
    std::vector<size_t> B = Matrix(B_ROWS,B_COLS);
    std::vector<size_t> C1 = std::vector<size_t> (A_ROWS * B_COLS,0);
    std::vector<size_t> C2 = std::vector<size_t> (A_ROWS * B_COLS,0);

    double timeEfficientAccess = Time([&]() { EfficientMatrixesMultiplier(C1, A, B); });
    results << "Tiempo de Multiplicacion en Eficiente:" << timeEfficientAccess << std::endl;
    // Llamada a PrintTime con lambda para NonEfficientMatrixesMultiplier
    double timeNonEfficientAccess = Time([&]() { NonEfficientMatrixesMultiplier(C1, A, B); });
    results << "Tiempo de Multiplicacion en No Eficiente:" << timeNonEfficientAccess << std::endl;
    
    results.close();

    return 0;
}
