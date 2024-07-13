#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
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
PGMImage readPGM(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("No se pudo abrir el archivo: " + filename);
    }

    PGMImage img;
    std::string line;
    std::getline(file, line);
    if (line != "P5") {
        throw std::runtime_error("Formato de archivo no soportado. Solo se admite PGM binario (P5).");
    }

    // Saltar comentarios
    while (std::getline(file, line)) {
        if (line[0] != '#') break;
    }

    std::istringstream iss(line);
    iss >> img.width >> img.height;
    file >> img.max_val;
    file.ignore(); // Saltar el carácter de nueva línea

    img.data.resize(img.width * img.height);
    file.read(reinterpret_cast<char*>(img.data.data()), img.data.size());

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

// Función para ordenar una ventana de píxeles
__device__ void sortWindow(unsigned char* window, int windowSize) {
    for (int i = 0; i < windowSize * windowSize - 1; i++) {
        for (int j = 0; j < windowSize * windowSize - i - 1; j++) {
            if (window[j] > window[j + 1]) {
                unsigned char temp = window[j];
                window[j] = window[j + 1];
                window[j + 1] = temp;
            }
        }
    }
}

// Kernel para aplicar el filtro de mediana
__global__ void medianFilterKernel(unsigned char* input, unsigned char* output, int width, int height, int windowSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        unsigned char window[81]; // Asumimos un tamaño máximo de ventana de 5x5
        int idx = 0;
        int halfWindow = windowSize / 2;

        for (int wy = -halfWindow; wy <= halfWindow; wy++) {
            for (int wx = -halfWindow; wx <= halfWindow; wx++) {
                int nx = x + wx;
                int ny = y + wy;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    window[idx++] = input[ny * width + nx];
                } else {
                    window[idx++] = 0; // Padding con 0 para píxeles fuera de la imagen
                }
            }
        }
        sortWindow(window, windowSize);
        output[y * width + x] = window[(windowSize * windowSize) / 2]; // Selecciona la mediana
    }
}

// Función para aplicar el filtro de mediana en la GPU y medir el tiempo
float applyMedianFilterGPU(const PGMImage& input, PGMImage& output, int windowSize) {
    unsigned char *d_input, *d_output;
    size_t size = input.width * input.height * sizeof(unsigned char);

    // Asignar memoria en la GPU
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copiar datos de entrada a la GPU
    cudaMemcpy(d_input, input.data.data(), size, cudaMemcpyHostToDevice);

    // Configurar la ejecución del kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((input.width + blockSize.x - 1) / blockSize.x, (input.height + blockSize.y - 1) / blockSize.y);

    // Crear eventos para medir el tiempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Iniciar el temporizador
    cudaEventRecord(start);

    // Lanzar el kernel
    medianFilterKernel<<<gridSize, blockSize>>>(d_input, d_output, input.width, input.height, windowSize);

    // Detener el temporizador
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calcular el tiempo transcurrido
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copiar resultados de vuelta a la CPU
    cudaMemcpy(output.data.data(), d_output, size, cudaMemcpyDeviceToHost);

    // Liberar memoria y destruir eventos
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
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
        PGMImage filtered = img; // Inicializar con la misma estructura

        const int NUM_ITERATIONS = 10;
        std::vector<float> times(NUM_ITERATIONS);

        for (int i = 0; i < NUM_ITERATIONS; ++i) {
            times[i] = applyMedianFilterGPU(img, filtered, windowSize);
        }

        // Calcular media
        float mean = std::accumulate(times.begin(), times.end(), 0.0f) / NUM_ITERATIONS;

        // Calcular desviación estándar
        float sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0f);
        float stdev = std::sqrt(sq_sum / NUM_ITERATIONS - mean * mean);

        std::cout << "Tiempo promedio: " << mean << " ms" << std::endl;
        std::cout << "Desviación estándar: " << stdev << " ms" << std::endl;

        writePGM(outputFilename, filtered);
        std::cout << "Filtro mediana aplicado exitosamente. Resultado guardado en " << outputFilename << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}