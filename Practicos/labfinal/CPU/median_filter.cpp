#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

struct PGMImage {
    int width;
    int height;
    int max_val;
    std::vector<unsigned char> data;
};

// Función para leer una imagen PGM
PGMImage readPGM(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("No se pudo abrir el archivo.");
    }

    PGMImage img;
    std::string line;
    std::getline(file, line);
    if (line != "P5" && line != "P2") {
        throw std::runtime_error("Formato de archivo no soportado. Solo se admite PGM binario (P5) o ASCII (P2).");
    }

    bool isBinary = (line == "P5");

    // Saltar comentarios
    while (std::getline(file, line)) {
        if (line[0] != '#') break;
    }

    std::istringstream iss(line);
    iss >> img.width >> img.height;
    file >> img.max_val;
    file.ignore(); // Saltar el carácter de nueva línea

    img.data.resize(img.width * img.height);
    if (isBinary) {
        file.read(reinterpret_cast<char*>(img.data.data()), img.data.size());
    } else {
        for (int i = 0; i < img.width * img.height; ++i) {
            int pixel;
            file >> pixel;
            img.data[i] = static_cast<unsigned char>(pixel);
        }
    }

    return img;
}

// Función para escribir una imagen PGM
void writePGM(const std::string& filename, const PGMImage& img) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("No se pudo crear el archivo: " + filename);
    }

    file << "P5\n" << img.width << " " << img.height << "\n" << img.max_val << "\n";
    file.write(reinterpret_cast<const char*>(img.data.data()), img.data.size());
}
// Función para aplicar el filtro mediana a un pixel
unsigned char medianFilter(const PGMImage& img, int x, int y, int windowSize) {
    std::vector<unsigned char> window;
    int halfWindow = windowSize / 2;

    for (int wy = -halfWindow; wy <= halfWindow; ++wy) {
        for (int wx = -halfWindow; wx <= halfWindow; ++wx) {
            int nx = x + wx;
            int ny = y + wy;
            
            if (nx < 0 || nx >= img.width || ny < 0 || ny >= img.height) {
                window.push_back(0);
            } else {
                window.push_back(img.data[ny * img.width + nx]);
            }
        }
    }

    std::sort(window.begin(), window.end());
    return window[window.size() / 2];
}

// Función principal para aplicar el filtro mediana a toda la imagen
PGMImage applyMedianFilter(const PGMImage& input, int windowSize) {
    PGMImage output = input;
    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            output.data[y * input.width + x] = medianFilter(input, x, y, windowSize);
        }
    }
    return output;
}
// Función para calcular la media
double calculateMean(const std::vector<double>& times) {
    return std::accumulate(times.begin(), times.end(), 0.0) / times.size();
}

// Función para calcular la desviación estándar
double calculateStdDev(const std::vector<double>& times, double mean) {
    double squareSum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
    return std::sqrt(squareSum / times.size() - mean * mean);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Uso: " << argv[0] << " <imagen_entrada.pgm> <imagen_salida.pgm> <tamaño_ventana>" << std::endl;
        return 1;
    }

    const char* inputFilename = argv[1];
    const char* outputFilename = argv[2];
    int windowSize = std::atoi(argv[3]);

    if (windowSize % 2 == 0) {
        std::cerr << "El tamaño de la ventana debe ser impar." << std::endl;
        return 1;
    }

    try {
        PGMImage img = readPGM(inputFilename);
        std::vector<double> executionTimes;

        for (int i = 0; i < 10; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            
            PGMImage filtered = applyMedianFilter(img, windowSize);
            
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            executionTimes.push_back(duration.count());

            if (i == 0) {
                writePGM(outputFilename, filtered);
            }
        }

        double mean = calculateMean(executionTimes);
        double stdDev = calculateStdDev(executionTimes, mean);

        std::cout << "Filtro mediana aplicado exitosamente. Resultado guardado en " << outputFilename << std::endl;
        std::cout << "Estadísticas de tiempo de ejecución (ms) para 10 ejecuciones:" << std::endl;
        std::cout << "Media: " << mean << std::endl;
        std::cout << "Desviación estándar: " << stdDev << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}