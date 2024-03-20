#include <iostream>
#include <vector>
#include <chrono>

void coolDownCache() {
    std::vector<int> cooldownData(1024 * 1024 * 256, 1);  // Tamaño suficiente para sobrepasar la caché L3
    long long dummySum = 0;
    for (size_t i = 0; i < cooldownData.size(); ++i) {
        dummySum += cooldownData[i];
    }
    // Usa dummySum de alguna manera para evitar que el compilador optimice el bucle
    if (dummySum == -1) {
        std::cout << "This won't happen." << std::endl;
    }
}

int main() {
    const size_t size = 1024 * 1024 * 256; 
    std::vector<int> data(size, 1);


    // Acceso disperso
    auto start = std::chrono::high_resolution_clock::now();
    long long sum2 = 0;
    for (size_t i = 0; i < data.size(); i += 16) {
        for (size_t j = 0; j < 16; j++) {
            sum2 += data[i + j];
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double>  diff = end - start;
    std::cout << "Tiempo acceso disperso: " << diff.count() << " s" << std::endl;
    
    coolDownCache();  // Enfriamiento de caché
    
    // Acceso secuencial
    start = std::chrono::high_resolution_clock::now();
    long long sum1 = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        sum1 += data[i];
    }
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "Tiempo acceso secuencial: " << diff.count() << " s" << std::endl;

    return 0;
}
