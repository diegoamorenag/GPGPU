#include <iostream>
#include <vector>
#include <chrono>
#include "../auxFunctions.h"

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

long long DiserseAccess(std::vector<int> data, const size_t size){
    long long sum = 0;
    for (size_t i = 0; i < size; i += 16) {
        for (size_t j = 0; j < 16; j++) {
            sum += data[i + j];
        }
    }
    return sum;
}

long long SecuentialAccess(std::vector<int> data, const size_t size){
    long long sum = 0;
    for (size_t i = 0; i < size; i ++) {
        sum += data[i];
    }
    return sum;
}

int main() {
    const size_t size = 1024 * 1024 * 256; 
    std::vector<int> data(size, 1);

    double timeDisperseAccess = Time([&]() { DiserseAccess(data, size); });
    std::cout << "Tiempo acceso disperso: " << timeDisperseAccess << " s" << std::endl;
    
    coolDownCache();  // Enfriamiento de caché
    
    double timeSecuentialAccess = Time([&]() { SecuentialAccess(data, size); });
    std::cout << "Tiempo acceso secuencial: " << timeSecuentialAccess << " s" << std::endl;

    return 0;
}
