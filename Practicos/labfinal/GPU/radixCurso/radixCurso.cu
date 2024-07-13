#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace std;

struct PGMImage {
    int width;
    int height;
    int max_val;
    std::vector<unsigned char> data;
};

PGMImage readPGM(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("No se pudo abrir el archivo.");
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
void writePGM(const char* filename, const PGMImage& img) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error(std::string("No se pudo crear el archivo: ") + filename);
    }

    file << "P5\n" << img.width << " " << img.height << "\n" << img.max_val << "\n";
    file.write(reinterpret_cast<const char*>(img.data.data()), img.data.size());
}

__device__ unsigned int getBit(unsigned char value, int bitPosition) {
    return (value >> bitPosition) & 1;
}

__global__ void computeFlags(unsigned char* input, int* flags, int n, int bitPosition) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        flags[idx] = !getBit(input[idx], bitPosition);
    }
}

__global__ void radixSortStep(unsigned char* input, unsigned char* output, int* prefixSum, int n, int bitPosition) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned int bit = getBit(input[idx], bitPosition);
        int position;
        if (bit == 0) {
            position = prefixSum[idx];
        } else {
            position = idx - prefixSum[idx] + prefixSum[n-1];
        }
        output[position] = input[idx];
    }
}

__global__ void checkIfSorted(unsigned char* input, int* isSorted, int n) {
    __shared__ int localIsSorted;
    if (threadIdx.x == 0) {
        localIsSorted = 1;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n - 1) {
        if (input[idx] > input[idx + 1]) {
            atomicAnd(&localIsSorted, 0);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAnd(isSorted, localIsSorted);
    }
}

__device__ void radixSort(unsigned char* input, unsigned char* output, int n) {
    // Variables locales para el dispositivo
    int flags[256];  // Asumimos un tamaño máximo de ventana de 16x16
    int prefixSum[256];
    
    for (int bit = 0; bit < 8; ++bit) {
        // Compute flags
        for (int i = 0; i < n; ++i) {
            flags[i] = !getBit(input[i], bit);
        }
        
        // Compute prefix sum (simple implementation for device)
        prefixSum[0] = 0;
        for (int i = 1; i < n; ++i) {
            prefixSum[i] = prefixSum[i-1] + flags[i-1];
        }
        
        // Perform radix sort step
        for (int i = 0; i < n; ++i) {
            unsigned int b = getBit(input[i], bit);
            int position;
            if (b == 0) {
                position = prefixSum[i];
            } else {
                position = i - prefixSum[i] + prefixSum[n-1];
            }
            output[position] = input[i];
        }
        
        // Swap input and output
        unsigned char* temp = input;
        input = output;
        output = temp;
        
        // Check if sorted (simple implementation for device)
        bool isSorted = true;
        for (int i = 0; i < n - 1; ++i) {
            if (input[i] > input[i + 1]) {
                isSorted = false;
                break;
            }
        }
        if (isSorted) break;
    }
    
    // Ensure the result is in the output array
    if (input != output) {
        for (int i = 0; i < n; ++i) {
            output[i] = input[i];
        }
    }
}
// Kernel for applying median filter using shared memory and radix sort
template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int WINDOW_SIZE>
__global__ void medianFilterRadixSortKernel(unsigned char* input, unsigned char* output, int width, int height) {
    __shared__ unsigned char sharedMem[BLOCK_DIM_Y + WINDOW_SIZE - 1][BLOCK_DIM_X + WINDOW_SIZE - 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * BLOCK_DIM_X;
    int by = blockIdx.y * BLOCK_DIM_Y;
    int x = bx + tx;
    int y = by + ty;

    // Cargar datos en memoria compartida (igual que antes)
    for (int dy = ty; dy < BLOCK_DIM_Y + WINDOW_SIZE - 1; dy += BLOCK_DIM_Y) {
        for (int dx = tx; dx < BLOCK_DIM_X + WINDOW_SIZE - 1; dx += BLOCK_DIM_X) {
            int globalX = bx + dx - WINDOW_SIZE / 2;
            int globalY = by + dy - WINDOW_SIZE / 2;

            if (globalX >= 0 && globalX < width && globalY >= 0 && globalY < height) {
                sharedMem[dy][dx] = input[globalY * width + globalX];
            } else {
                sharedMem[dy][dx] = 0;
            }
        }
    }

    __syncthreads();

    // Aplicar el filtro de mediana usando Radix Sort
    if (x < width && y < height) {
        unsigned char window[WINDOW_SIZE * WINDOW_SIZE];
        int idx = 0;

        for (int wy = 0; wy < WINDOW_SIZE; wy++) {
            for (int wx = 0; wx < WINDOW_SIZE; wx++) {
                window[idx++] = sharedMem[ty + wy][tx + wx];
            }
        }

        // Aplicar Radix Sort a la ventana
        unsigned char sortedWindow[WINDOW_SIZE * WINDOW_SIZE];
        radixSort(window, sortedWindow, WINDOW_SIZE * WINDOW_SIZE);

        output[y * width + x] = sortedWindow[(WINDOW_SIZE * WINDOW_SIZE) / 2];
    }
}

// Function to apply median filter on GPU and measure time
float applyMedianFilterGPU(const PGMImage& input, PGMImage& output, int windowSize) {
    unsigned char *d_input, *d_output;
    size_t size = input.width * input.height * sizeof(unsigned char);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, input.data.data(), size, cudaMemcpyHostToDevice);

    const int BLOCK_DIM_X = 16;
    const int BLOCK_DIM_Y = 16;
    dim3 blockSize(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridSize((input.width + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (input.height + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Lanzar el kernel apropiado según el tamaño de la ventana
    switch (windowSize) {
        case 3:
            medianFilterRadixSortKernel<BLOCK_DIM_X, BLOCK_DIM_Y, 3><<<gridSize, blockSize>>>(d_input, d_output, input.width, input.height);
            break;
        case 5:
            medianFilterRadixSortKernel<BLOCK_DIM_X, BLOCK_DIM_Y, 5><<<gridSize, blockSize>>>(d_input, d_output, input.width, input.height);
            break;
        case 7:
            medianFilterRadixSortKernel<BLOCK_DIM_X, BLOCK_DIM_Y, 7><<<gridSize, blockSize>>>(d_input, d_output, input.width, input.height);
            break;
        case 9:
            medianFilterRadixSortKernel<BLOCK_DIM_X, BLOCK_DIM_Y, 9><<<gridSize, blockSize>>>(d_input, d_output, input.width, input.height);
            break;
        case 11:
            medianFilterRadixSortKernel<BLOCK_DIM_X, BLOCK_DIM_Y, 11><<<gridSize, blockSize>>>(d_input, d_output, input.width, input.height);
            break;
        default:
            throw std::runtime_error("Tamaño de ventana no soportado");
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
        std::cerr << "El tamaño de la ventana debe ser impar." << std::endl;
        return 1;
    }

    try {
        PGMImage img = readPGM(inputFilename);
        PGMImage filtered = img; // Inicializar con la misma estructura

        const int NUM_ITERATIONS = 100;
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