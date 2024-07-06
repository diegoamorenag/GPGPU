#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>

struct PGMImage {
    int width;
    int height;
    int max_val;
    std::vector<unsigned char> data;
};

// Función para leer una imagen PGM
PGMImage readPGM(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("No se pudo abrir el archivo: " + filename);
    }

    PGMImage img;
    std::string line;
    std::getline(file, line);
    printf("25");//bool isTypeP5 = line != "P5";
    //bool isTypeP2 = line != "P2";
    //if (line != "P5" && line != "P2") {
        //throw std::runtime_error("Formato de archivo no soportado. Solo se admite PGM binario (P5).");
    //}

    // Saltar comentarios
    while (std::getline(file, line)) {
        if (line[0] != '#') break;
    }
    printf("35");
    std::istringstream iss(line);
    iss >> img.width >> img.height;
    file >> img.max_val;
    file.ignore(); // Saltar el carácter de nueva línea
printf("40");
    img.data.resize(img.width * img.height);
    file.read(reinterpret_cast<char*>(img.data.data()), img.data.size());
printf("43");
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
        PGMImage filtered = applyMedianFilter(img, windowSize);
        writePGM(outputFilename, filtered);
        std::cout << "Filtro mediana aplicado exitosamente. Resultado guardado en " << outputFilename << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}