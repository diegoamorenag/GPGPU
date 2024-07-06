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

// Declaración de la textura
texture<unsigned char, 2, cudaReadModeElementType> texInput;

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


template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int WINDOW_SIZE>
__global__ void medianFilterOptimizedKernel(unsigned char* output, int width, int height) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * BLOCK_DIM_X + tx;
    int y = blockIdx.y * BLOCK_DIM_Y + ty;

    if (x < width && y < height) {
        unsigned char window[WINDOW_SIZE * WINDOW_SIZE];
        int idx = 0;

        for (int wy = -WINDOW_SIZE/2; wy <= WINDOW_SIZE/2; wy++) {
            for (int wx = -WINDOW_SIZE/2; wx <= WINDOW_SIZE/2; wx++) {
                float u = x + wx + 0.5f;
                float v = y + wy + 0.5f;
                window[idx++] = tex2D(texInput, u, v);
            }
        }

        radixSort(window, WINDOW_SIZE * WINDOW_SIZE);
        output[y * width + x] = window[(WINDOW_SIZE * WINDOW_SIZE) / 2];
    }
}

// Function to apply median filter on GPU and measure time
float applyMedianFilterGPU(const PGMImage& input, PGMImage& output, int windowSize) {
    unsigned char *d_output;
    size_t size = input.width * input.height * sizeof(unsigned char);

    // Allocate CUDA array and copy input data
    cudaArray* cuArray;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
    cudaMallocArray(&cuArray, &channelDesc, input.width, input.height);
    cudaMemcpyToArray(cuArray, 0, 0, input.data.data(), size, cudaMemcpyHostToDevice);

    // Set texture parameters
    texInput.addressMode[0] = cudaAddressModeClamp;
    texInput.addressMode[1] = cudaAddressModeClamp;
    texInput.filterMode = cudaFilterModePoint;
    texInput.normalized = false;

    // Bind the array to the texture
    cudaBindTextureToArray(texInput, cuArray);

    cudaMalloc(&d_output, size);

    const int BLOCK_DIM_X = 16;
    const int BLOCK_DIM_Y = 16;
    dim3 blockSize(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridSize((input.width + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (input.height + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch appropriate kernel based on window size
    switch (windowSize) {
        case 3:
            medianFilterOptimizedKernel<BLOCK_DIM_X, BLOCK_DIM_Y, 3><<<gridSize, blockSize>>>(d_output, input.width, input.height);
            break;
        case 5:
            medianFilterOptimizedKernel<BLOCK_DIM_X, BLOCK_DIM_Y, 5><<<gridSize, blockSize>>>(d_output, input.width, input.height);
            break;
        case 7:
            medianFilterOptimizedKernel<BLOCK_DIM_X, BLOCK_DIM_Y, 7><<<gridSize, blockSize>>>(d_output, input.width, input.height);
            break;
        case 9:
            medianFilterOptimizedKernel<BLOCK_DIM_X, BLOCK_DIM_Y, 9><<<gridSize, blockSize>>>(d_output, input.width, input.height);
            break;
        case 11:
            medianFilterOptimizedKernel<BLOCK_DIM_X, BLOCK_DIM_Y, 11><<<gridSize, blockSize>>>(d_output, input.width, input.height);
            break;
        default:
            throw std::runtime_error("Unsupported window size");
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(output.data.data(), d_output, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaUnbindTexture(texInput);
    cudaFreeArray(cuArray);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1.0f;  // Indicate error
    }

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