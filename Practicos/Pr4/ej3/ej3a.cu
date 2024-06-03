#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>
#include <cmath>
#include <vector>

#define WIDTH 3840
#define HEIGHT 2160
#define HISTOGRAM_SIZE 256
#define BLOCK_SIZE 256
#define RUNS 10

#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void histograma_kernel(int *image, unsigned int *histogram, int width, int height) {
    __shared__ unsigned int local_histogram[HISTOGRAM_SIZE];

    int tid = threadIdx.x;
    if (tid < HISTOGRAM_SIZE) {
        local_histogram[tid] = 0;
    }
    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        atomicAdd(&local_histogram[image[idx]], 1);
    }
    __syncthreads();

    if (tid < HISTOGRAM_SIZE) {
        atomicAdd(&histogram[tid], local_histogram[tid]);
    }
}

void histograma_sequential(int *image, unsigned int *histogram, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        histogram[image[i]]++;
    }
}

double media(const std::vector<double>& times) {
    double sum = 0.0;
    for (double time : times) {
        sum += time;
    }
    return sum / times.size();
}

double desvio(const std::vector<double>& times, double mean) {
    double sum = 0.0;
    for (double time : times) {
        sum += (time - mean) * (time - mean);
    }
    return sqrt(sum / times.size());
}

int main() {
    int *h_image, *d_image;
    unsigned int *d_histogram, h_histogram[HISTOGRAM_SIZE];
    unsigned int h_histogram_seq[HISTOGRAM_SIZE];

    size_t imageSize = WIDTH * HEIGHT * sizeof(int);

    // inicializamos una imagen random en el host
    h_image = (int *)malloc(imageSize);
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        h_image[i] = rand() % 256;
    }

    // reservamos memoria en gpu
    CUDA_CHK(cudaMalloc((void**)&d_image, imageSize));
    CUDA_CHK(cudaMalloc((void**)&d_histogram, HISTOGRAM_SIZE * sizeof(unsigned int)));

    std::vector<double> parallel_times;
    std::vector<double> sequential_times;

    for (int run = 0; run < RUNS; run++) {
        CUDA_CHK(cudaMemset(d_histogram, 0, HISTOGRAM_SIZE * sizeof(unsigned int)));

        // coiamos la imagen a la gpu
        CUDA_CHK(cudaMemcpy(d_image, h_image, imageSize, cudaMemcpyHostToDevice));

        // lanzamos el kernel
        dim3 blockSize(BLOCK_SIZE);
        dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);

        auto start_time = std::chrono::high_resolution_clock::now();
        histograma_kernel<<<gridSize, blockSize>>>(d_image, d_histogram, WIDTH, HEIGHT);
        CUDA_CHK(cudaDeviceSynchronize());
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> parallel_time = end_time - start_time;
        parallel_times.push_back(parallel_time.count());

        // copiamos el histograma de la gpu al host
        CUDA_CHK(cudaMemcpy(h_histogram, d_histogram, HISTOGRAM_SIZE * sizeof(unsigned int), cudaMemcpyDeviceToHost));

        // implementacion secuencial
        for (int i = 0; i < HISTOGRAM_SIZE; i++) {
            h_histogram_seq[i] = 0;
        }

        start_time = std::chrono::high_resolution_clock::now();
        histograma_sequential(h_image, h_histogram_seq, WIDTH, HEIGHT);
        end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> sequential_time = end_time - start_time;
        sequential_times.push_back(sequential_time.count());
    }

    // calculamos media y desvio estandar
    double parallel_mean = media(parallel_times);
    double parallel_stddev = desvio(parallel_times, parallel_mean);

    double sequential_mean = media(sequential_times);
    double sequential_stddev = desvio(sequential_times, sequential_mean);

    // imprimir los histogramas para ver si coinciden solo en la ultima corrida, lo usamos solo para debbugear :)
    // printf("Histograma con Reduce paralelo:\n");
    // for (int i = 0; i < HISTOGRAM_SIZE; i++) {
    //     printf("Valor %d: %u ocurrencias\n", i, h_histogram[i]);
    // }

    // printf("Histograma secuencial:\n");
    // for (int i = 0; i < HISTOGRAM_SIZE; i++) {
    //     printf("Valor %d: %u ocurrencias\n", i, h_histogram_seq[i]);
    // }

    // comparar los histogramas
    bool correct = true;
    for (int i = 0; i < HISTOGRAM_SIZE; i++) {
        if (h_histogram[i] != h_histogram_seq[i]) {
            correct = false;
            printf("Diferencia en valor %d: paralelo %u, secuencial %u\n", i, h_histogram[i], h_histogram_seq[i]);
        }
    }

    if (correct) {
        printf("Los histogramas coinciden\n");
    } else {
        printf("Los histogramas no coinciden\n");
    }

    // mostramos lso tiempos
    printf("Tiempo de ejecucion en paralelo: media = %f ms, desvio = %f ms\n", parallel_mean*1000, parallel_stddev*1000);
    printf("Tiempo de ejecucion secuencial: media = %f ms, desvio = %f ms\n", sequential_mean*1000, sequential_stddev*1000);

    // liberamos memoria
    free(h_image);
    CUDA_CHK(cudaFree(d_image));
    CUDA_CHK(cudaFree(d_histogram));

    return 0;
}
