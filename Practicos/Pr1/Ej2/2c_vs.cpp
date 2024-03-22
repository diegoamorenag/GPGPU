#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include "../auxFunctions.h"

using namespace std;

#define L3_SIZE (16 * 1024 * 1024) // Tamaño de la caché L3 en bytes

std::vector<size_t> Matrix(size_t rows, size_t cols){
    std::vector<size_t> res(rows * cols,1);
    return res;
}

void EfficientMatrixesMultiplier(std::vector<size_t>& C, const std::vector<size_t>& A, const std::vector<size_t>& B, int size){
    for (size_t it = 0; it < size; it++) { 
        for (size_t row = 0; row < size; row++) {
            for (size_t col = 0; col < size; col++) {
                C[row * size + col] += A[row * size + it] * B[it * size + col];
            }
        }
    }
}

int main() {
    system("mkdir -p Ej2/results");

    std::ofstream results("Ej2/results/2c_vs");
    std::vector<int> sizes = {512, 1024, 2048};

    for (int size : sizes) {       
        std::vector<size_t> A = Matrix(size,size);
        std::vector<size_t> B = Matrix(size,size);
        std::vector<size_t> C1 = std::vector<size_t> (size * size,0);
        double timeEfficientAccess = Time([&]() { EfficientMatrixesMultiplier(C1, A, B, size); });
        results << "Tiempo de Multiplicacion en Eficiente:" << timeEfficientAccess << std::endl;
    }   
    results.close();

    return 0;
}
