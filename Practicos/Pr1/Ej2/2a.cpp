#include <iostream>
#include <vector>
#include <chrono>

typedef int VALT; // Usamos int en lugar de double

// Calcula GFLOPS
double gflops(int m, int n, int p, double seconds) {
    double operations = 2.0 * m * n * p;
    double gflops = operations / (seconds * 1e9);
    return gflops;
}

// Multiplicación de matrices original
void multiplyMatrices(const std::vector<VALT>& A, const std::vector<VALT>& B, std::vector<VALT>& C, int m, int n, int p) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            VALT sum = 0;
            for (int it = 0; it < p; ++it) {
                sum += A[row * p + it] * B[it * n + col];
            }
            C[row * n + col] = sum;
        }
    }
}

// Multiplicación de matrices optimizada
void multiplyMatricesOptimized(const std::vector<VALT>& A, const std::vector<VALT>& B, std::vector<VALT>& C, int m, int n, int p) {
    for (int it = 0; it < p; ++it) {
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
                C[row * n + col] += A[row * p + it] * B[it * n + col];
            }
        }
    }
}

int main() {
    // Prueba con diferentes tamaños de matrices
    std::vector<std::tuple<int, int, int>> sizes = {
        {1600, 1600, 1600},
        {3200, 3200, 3200},
        {6400, 6400, 6400},
    };

    for (auto& [m, n, p] : sizes) {
        // Inicializa matrices
        std::vector<VALT> A(m * p, 1);
        std::vector<VALT> B(p * n, 2);
        std::vector<VALT> C(m * n, 0);

        // Mide el tiempo de la versión original
        auto start = std::chrono::high_resolution_clock::now();
        multiplyMatrices(A, B, C, m, n, p);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Original (" << m << "x" << p << " y " << p << "x" << n << "): "
                  << gflops(m, n, p, elapsed.count()) << " GFLOPS" << std::endl;

        // Mide el tiempo de la versión optimizada
        std::fill(C.begin(), C.end(), 0); // Resetea C para la siguiente multiplicación
        start = std::chrono::high_resolution_clock::now();
        multiplyMatricesOptimized(A, B, C, m, n, p);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "Optimizado (" << m << "x" << p << " y " << p << "x" << n << "): "
                  << gflops(m, n, p, elapsed.count()) << " GFLOPS" << std::endl;
    }

    return 0;
}
