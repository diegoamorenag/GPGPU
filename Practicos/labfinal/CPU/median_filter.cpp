#include <iostream>
#include <vector>
#include <algorithm>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Función para aplicar el filtro mediana a un pixel
unsigned char medianFilter(unsigned char* img, int width, int height, int x, int y, int windowSize) {
    std::vector<unsigned char> window;
    int halfWindow = windowSize / 2;

    for (int wy = -halfWindow; wy <= halfWindow; ++wy) {
        for (int wx = -halfWindow; wx <= halfWindow; ++wx) {
            int nx = x + wx;
            int ny = y + wy;
            
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                window.push_back(0);
            } else {
                window.push_back(img[ny * width + nx]);
            }
        }
    }

    std::sort(window.begin(), window.end());
    return window[window.size() / 2];
}

// Función principal para aplicar el filtro mediana a toda la imagen
void applyMedianFilter(unsigned char* input, unsigned char* output, int width, int height, int windowSize) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            output[y * width + x] = medianFilter(input, width, height, x, y, windowSize);
        }
        std::cout << "Progreso: " << (y * 100) / height << "%" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Uso: " << argv[0] << " <imagen_entrada> <imagen_salida> <tamaño_ventana>" << std::endl;
        return 1;
    }

    const char* inputFilename = argv[1];
    const char* outputFilename = argv[2];
    int windowSize = std::atoi(argv[3]);

    if (windowSize % 2 == 0) {
        std::cerr << "El tamano de la ventana debe ser impar." << std::endl;
        return 1;
    }

    int width, height, channels;
    unsigned char* img = stbi_load(inputFilename, &width, &height, &channels, 1);

    if (!img) {
        std::cerr << "Error al cargar la imagen." << std::endl;
        return 1;
    }

    std::cout << "Imagen cargada exitosamente." << std::endl;
    std::cout << "Dimensiones: " << width << "x" << height << std::endl;

    unsigned char* output = new unsigned char[width * height];

    applyMedianFilter(img, output, width, height, windowSize);

    if (!stbi_write_png(outputFilename, width, height, 1, output, width)) {
        std::cerr << "Error al guardar la imagen." << std::endl;
        stbi_image_free(img);
        delete[] output;
        return 1;
    }

    std::cout << "Filtro mediana aplicado exitosamente. Resultado guardado en " << outputFilename << std::endl;

    stbi_image_free(img);
    delete[] output;

    return 0;
}