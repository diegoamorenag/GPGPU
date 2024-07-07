#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>

#include <cmath>
#include <numeric>
#include <algorithm>

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

// Radix sort implementation for unsigned char (8-bit integers)
__device__ void radixSort(unsigned char* arr, int n) {
    unsigned char output[256];  // Assuming window size is at most 16x16 = 256
    int count[256] = {0};

    // Count occurrences of each digit
    for (int i = 0; i < n; i++) {
        count[arr[i]]++;
    }

    // Compute cumulative count
    for (int i = 1; i < 256; i++) {
        count[i] += count[i - 1];
    }

    // Build the output array
    for (int i = n - 1; i >= 0; i--) {
        output[count[arr[i]] - 1] = arr[i];
        count[arr[i]]--;
    }

    // Copy the output array to original array
    for (int i = 0; i < n; i++) {
        arr[i] = output[i];
    }
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

void radixSort(unsigned char* d_input, unsigned char* d_output, int n) {
    int* d_flags;
    int* d_prefixSum;
    int* d_isSorted;
    cudaMalloc(&d_flags, n * sizeof(int));
    cudaMalloc(&d_prefixSum, n * sizeof(int));
    cudaMalloc(&d_isSorted, sizeof(int));

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    for (int bit = 0; bit < 8; ++bit) {
        computeFlags<<<gridSize, blockSize>>>(d_input, d_flags, n, bit);
        thrust::exclusive_scan(thrust::device, d_flags, d_flags + n, d_prefixSum);

        radixSortStep<<<gridSize, blockSize>>>(d_input, d_output, d_prefixSum, n, bit);

        // Swap input and output
        unsigned char* temp = d_input;
        d_input = d_output;
        d_output = temp;

        // Check if sorted
        int isSorted = 1;
        cudaMemcpy(d_isSorted, &isSorted, sizeof(int), cudaMemcpyHostToDevice);
        checkIfSorted<<<gridSize, blockSize>>>(d_input, d_isSorted, n);
        cudaMemcpy(&isSorted, d_isSorted, sizeof(int), cudaMemcpyDeviceToHost);
        if (isSorted) break;
    }

    cudaFree(d_flags);
    cudaFree(d_prefixSum);
    cudaFree(d_isSorted);
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