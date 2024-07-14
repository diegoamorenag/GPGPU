#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

using namespace std;

struct PGMImage {
    int width;
    int height;
    int max_val;
    std::vector<unsigned char> data;
};

// Funcion para leer una imagen PGM
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

// Funcion para escribir una imagen PGM
void writePGM(const std::string& filename, const PGMImage& img) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("No se pudo crear el archivo: " + filename);
    }

    file << "P5\n" << img.width << " " << img.height << "\n" << img.max_val << "\n";
    file.write(reinterpret_cast<const char*>(img.data.data()), img.data.size());
}

template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int WINDOW_SIZE>
__global__ void medianFilterSharedKernel(unsigned char* input, unsigned char* output, int width, int height) {
    __shared__ unsigned char sharedMem[(BLOCK_DIM_Y + WINDOW_SIZE - 1) * (BLOCK_DIM_X + WINDOW_SIZE - 1)];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * BLOCK_DIM_X;
    int by = blockIdx.y * BLOCK_DIM_Y;
    int x = bx + tx;
    int y = by + ty;

    int sharedWidth = BLOCK_DIM_X + WINDOW_SIZE - 1;
    int sharedHeight = BLOCK_DIM_Y + WINDOW_SIZE - 1;

    // Cargar datos en memoria compartida fila por fila
    for (int row = ty; row < sharedHeight; row += BLOCK_DIM_Y) {
        for (int col = tx; col < sharedWidth; col += BLOCK_DIM_X) {
            int globalX = bx + col - WINDOW_SIZE / 2;
            int globalY = by + row - WINDOW_SIZE / 2;

            if (globalX >= 0 && globalX < width && globalY >= 0 && globalY < height) {
                sharedMem[row * sharedWidth + col] = input[globalY * width + globalX];
            } else {
                sharedMem[row * sharedWidth + col] = 0;
            }
        }
    }
    __syncthreads();

    // Ordenar y encontrar la mediana usando thrust
    if (x < width && y < height) {
        unsigned char window[WINDOW_SIZE * WINDOW_SIZE];
        int idx = 0;
        for (int wy = 0; wy < WINDOW_SIZE; wy++) {
            for (int wx = 0; wx < WINDOW_SIZE; wx++) {
                window[idx++] = sharedMem[(ty + wy) * sharedWidth + (tx + wx)];
            }
        }

        thrust::sort(thrust::seq, window, window + WINDOW_SIZE * WINDOW_SIZE);
        output[y * width + x] = window[(WINDOW_SIZE * WINDOW_SIZE) / 2];  // Mediana
    }
}

// Funcion para aplicar el filtro de mediana en la GPU y medir el tiempo
float applyMedianFilterGPU(const PGMImage& input, PGMImage& output, int windowSize) {
    unsigned char *d_input, *d_output;
    size_t size = input.width * input.height * sizeof(unsigned char);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, input.data.data(), size, cudaMemcpyHostToDevice);

    const int BLOCK_DIM_X = 32;
    const int BLOCK_DIM_Y = 32;
    dim3 blockSize(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridSize((input.width + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (input.height + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Lanzar el kernel apropiado según el tamaño de la ventana
    switch (windowSize) {
        case 3:
            medianFilterSharedKernel<BLOCK_DIM_X, BLOCK_DIM_Y, 3><<<gridSize, blockSize>>>(d_input, d_output, input.width, input.height);
            break;
        case 5:
            medianFilterSharedKernel<BLOCK_DIM_X, BLOCK_DIM_Y, 5><<<gridSize, blockSize>>>(d_input, d_output, input.width, input.height);
            break;
        case 7:
            medianFilterSharedKernel<BLOCK_DIM_X, BLOCK_DIM_Y, 7><<<gridSize, blockSize>>>(d_input, d_output, input.width, input.height);
            break;
        case 9:
            medianFilterSharedKernel<BLOCK_DIM_X, BLOCK_DIM_Y, 9><<<gridSize, blockSize>>>(d_input, d_output, input.width, input.height);
            break;
        case 11:
            medianFilterSharedKernel<BLOCK_DIM_X, BLOCK_DIM_Y, 11><<<gridSize, blockSize>>>(d_input, d_output, input.width, input.height);
            break;
        default:
            throw std::runtime_error("Tamano de ventana no soportado");
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(output.data.data(), d_output, size, cudaMemcpyDeviceToHost);

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
        std::cerr << "El tamano de la ventana debe ser impar." << std::endl;
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
        std::cout << "Desviacion estandar: " << stdev << " ms" << std::endl;

        writePGM(outputFilename, filtered);
        std::cout << "Filtro mediana aplicado exitosamente. Resultado guardado en " << outputFilename << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}