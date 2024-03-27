#include <iostream>
#include <cstdlib>
#include <fstream>
#include <tuple>

#include "../auxFunctions.h"

#define CACHE_LINE_SIZE 64
#define L1_SIZE (32 * 1024) // Tamaño de la caché L1 para datos
#define L2_SIZE (256 * 1024) // Tamaño de la caché L2
#define L3_SIZE (12 * 1024 * 1024) // Tamaño de la caché L3
#define L1_TOTAL_LINES (L1_SIZE / CACHE_LINE_SIZE)
#define L2_TOTAL_LINES (L2_SIZE / CACHE_LINE_SIZE)
#define L3_TOTAL_LINES (L3_SIZE / CACHE_LINE_SIZE)

void access_cache(char *array, int size) {
    for (int i = 0; i < size; i += CACHE_LINE_SIZE) {
        array[i] = 1;
    }
}

void fill_cache(char *array, int size) {
    for (int i = 0; i < size; i ++) {
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
    system("mkdir -p Ej1/results");

    // Abrir el archivo en el modo de escritura
    std::ofstream results("Ej1/results/1a");

    auto [cpu_time_usedL1, line_time_usedL1] = test_cache(L1_SIZE/sizeof(char), L1_TOTAL_LINES/sizeof(char));
    results << "Tiempo para L1 Data Cache: " << cpu_time_usedL1 << " segundos\n";
    results << "Tiempo por linea L1: " << line_time_usedL1 << " segundos\n";

    auto [cpu_time_usedL2, line_time_usedL2] = test_cache(L2_SIZE/sizeof(char), L2_TOTAL_LINES/sizeof(char));
    results << "Tiempo para L2 Cache: " << cpu_time_usedL2 << " segundos\n";
    results << "Tiempo por linea L2: " << line_time_usedL2 << " segundos\n";

    auto [cpu_time_usedL3, line_time_usedL3] = test_cache(L3_SIZE/sizeof(char), L3_TOTAL_LINES/sizeof(char));
    results << "Tiempo para L3 Cache: " << cpu_time_usedL3 << " segundos\n";
    results << "Tiempo por linea L3: " << line_time_usedL3 << " segundos\n";

    results << "Relacion linea con cache 1 de cache 2: " << line_time_usedL2 / line_time_usedL1 << std::endl;
    results << "Relacion linea con cache 1 de cache 3: " << line_time_usedL3 / line_time_usedL1 << std::endl;

    results.close();
    return 0;
}
