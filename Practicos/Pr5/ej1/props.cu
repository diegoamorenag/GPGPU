#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Error obteniendo el número de dispositivos CUDA: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        err = cudaGetDeviceProperties(&deviceProp, device);
        if (err != cudaSuccess) {
            std::cerr << "Error obteniendo las propiedades del dispositivo: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }

        std::cout << "Propiedades del dispositivo " << device << ":" << std::endl;
        std::cout << "  Nombre: " << deviceProp.name << std::endl;
        std::cout << "  Memoria compartida por bloque: " << deviceProp.sharedMemPerBlock << " bytes" << std::endl;
        std::cout << "  Memoria compartida por multiprocesador: " << deviceProp.sharedMemPerMultiprocessor << " bytes" << std::endl;
        std::cout << "  Número de multiprocesadores: " << deviceProp.multiProcessorCount << std::endl;
    }

    return 0;
}