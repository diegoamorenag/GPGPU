#include <iostream>
#include <cstdlib>
#include <ctime>
#include <tuple>
#include "../auxFunctions.h"

#define CACHE_LINE_SIZE 64
#define L1_SIZE (192 * 1024) // Tamaño de la caché L1 para datos
#define L2_SIZE (3 * 1024 * 1024) // Tamaño de la caché L2
#define L3_SIZE (16 * 1024 * 1024) // Tamaño de la caché L3
#define L1_TOTAL_LINES (L1_SIZE / CACHE_LINE_SIZE) // 3072
#define L2_TOTAL_LINES (L2_SIZE / CACHE_LINE_SIZE) // 49152
#define L3_TOTAL_LINES (L3_SIZE / CACHE_LINE_SIZE) // 262144

void access_cache(char *array, int size) {
    for (int i = 0; i < size; i += CACHE_LINE_SIZE) {
        array[i] += 1;
    }
}

void fill_cache(char *array, int size) {
    for (int i = 0; i < size; i += CACHE_LINE_SIZE) {
        array[i] = 0;
    }
}

std::tuple<double, double> test_cache(int cache_size, int total_lines) {
    char *array = new char[cache_size];
    fill_cache(array, cache_size);
    
    double cpu_time_used = Time([&]() { access_cache(array, cache_size); });
    
    double line_time_used = cpu_time_used / total_lines;
    delete[] array;
    return std::make_tuple(cpu_time_used, line_time_used);
}

int main() {
    auto [cpu_time_usedL1, line_time_usedL1] = test_cache(L1_SIZE, L1_TOTAL_LINES);
    std::cout << "Tiempo para L1 Data Cache: " << cpu_time_usedL1 << " segundos\n";
    std::cout << "Tiempo por linea: " << line_time_usedL1 << " segundos\n";

    auto [cpu_time_usedL2, line_time_usedL2] = test_cache(L2_SIZE, L2_TOTAL_LINES);
    std::cout << "Tiempo para L2 Cache: " << cpu_time_usedL2 << " segundos\n";
    std::cout << "Tiempo por linea: " << line_time_usedL2 << " segundos\n";

    auto [cpu_time_usedL3, line_time_usedL3] = test_cache(L3_SIZE, L3_TOTAL_LINES);
    std::cout << "Tiempo para L3 Cache: " << cpu_time_usedL3 << " segundos\n";
    std::cout << "Tiempo por linea: " << line_time_usedL3 << " segundos\n";

    std::cout << "Relacion con cache 1 de cache 2: " << cpu_time_usedL2 / cpu_time_usedL1 << std::endl;
    std::cout << "Relacion linea con cache 1 de cache 2: " << line_time_usedL2 / line_time_usedL1 << std::endl;
    std::cout << "Relacion con cache 1 de cache 3: " << cpu_time_usedL3 / cpu_time_usedL1 << std::endl;
    std::cout << "Relacion linea con cache 1 de cache 3: " << line_time_usedL3 / line_time_usedL1 << std::endl;

    return 0;
}
