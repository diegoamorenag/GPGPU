#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>
#include <cmath>
#include <vector>

#define WIDTH 3840
#define HEIGHT 2160
#define HISTOGRAM_SIZE 256
#define BLOCK_SIZE 16
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

__global__ void kernel_histograma(int *image, unsigned int *histogram_matrix, int width, int height) {
    __shared__ unsigned int local_histogram[HISTOGRAM_SIZE];

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int block_id = blockIdx.x + blockIdx.y * gridDim.x;

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
        histogram_matrix[block_id * HISTOGRAM_SIZE + tid] = local_histogram[tid];
    }
}

__global__ void kernel_reduccion(unsigned int *histogram_matrix, unsigned int *intermediate, int num_blocks, int step) {
    extern __shared__ unsigned int intermedio[];
    int tid = threadIdx.x;
    int col_id = blockIdx.x;

    intermedio[tid] = 0;

    for (int i = tid; i < num_blocks; i += blockDim.x) {
        intermedio[tid] += histogram_matrix[i * HISTOGRAM_SIZE + col_id];
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            intermedio[tid] += intermedio[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        intermediate[col_id] = intermedio[0];
    }
}

void reducir_wrapper(unsigned int *d_histogram_matrix, unsigned int *d_global_histogram, int num_blocks) {
    int step = 1;
    unsigned int *d_intermediate;

    while (num_blocks > 1) {
        int blocks = (num_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
        CUDA_CHK(cudaMalloc((void**)&d_intermediate, blocks * HISTOGRAM_SIZE * sizeof(unsigned int)));
        
        kernel_reduccion<<<HISTOGRAM_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(unsigned int)>>>(d_histogram_matrix, d_intermediate, num_blocks, step);
        CUDA_CHK(cudaDeviceSynchronize());
        
        cudaMemcpy(d_histogram_matrix, d_intermediate, blocks * HISTOGRAM_SIZE * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
        cudaFree(d_intermediate);
        
        num_blocks = blocks;
        step *= 2;
    }

    cudaMemcpy(d_global_histogram, d_histogram_matrix, HISTOGRAM_SIZE * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
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
    unsigned int *d_histogram_matrix, *h_histogram_matrix, *d_global_histogram;
    unsigned int h_histogram[HISTOGRAM_SIZE];
    unsigned int h_histogram_seq[HISTOGRAM_SIZE];

    size_t imageSize = WIDTH * HEIGHT * sizeof(int);
    int num_blocks = ((WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE) * ((HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE);
    size_t histogramMatrixSize = num_blocks * HISTOGRAM_SIZE * sizeof(unsigned int);

    // Inicializamos una imagen random en el host
    h_image = (int *)malloc(imageSize);
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        h_image[i] = rand() % 256;
    }

    h_histogram_matrix = (unsigned int *)malloc(histogramMatrixSize);

    // Reservamos memoria en GPU
    CUDA_CHK(cudaMalloc((void**)&d_image, imageSize));
    CUDA_CHK(cudaMalloc((void**)&d_histogram_matrix, histogramMatrixSize));
    CUDA_CHK(cudaMalloc((void**)&d_global_histogram, HISTOGRAM_SIZE * sizeof(unsigned int)));

    std::vector<double> parallel_times;
    std::vector<double> sequential_times;

    for (int run = 0; run < RUNS; run++) {
        CUDA_CHK(cudaMemset(d_histogram_matrix, 0, histogramMatrixSize));
        CUDA_CHK(cudaMemcpy(d_image, h_image, imageSize, cudaMemcpyHostToDevice));

        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);

        auto start_time = std::chrono::high_resolution_clock::now();
        kernel_histograma<<<gridSize, blockSize>>>(d_image, d_histogram_matrix, WIDTH, HEIGHT);
        CUDA_CHK(cudaDeviceSynchronize());
        
        reducir_wrapper(d_histogram_matrix, d_global_histogram, num_blocks);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> parallel_time = end_time - start_time;
        parallel_times.push_back(parallel_time.count());

        CUDA_CHK(cudaMemcpy(h_histogram, d_global_histogram, HISTOGRAM_SIZE * sizeof(unsigned int), cudaMemcpyDeviceToHost));

        for (int i = 0; i < HISTOGRAM_SIZE; i++) {
            h_histogram_seq[i] = 0;
        }

        start_time = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < WIDTH * HEIGHT; i++) {
            h_histogram_seq[h_image[i]]++;
        }
        end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> sequential_time = end_time - start_time;
        sequential_times.push_back(sequential_time.count());

        bool correct = true;
        for (int i = 0; i < HISTOGRAM_SIZE; i++) {
            if (h_histogram[i] != h_histogram_seq[i]) {
                correct = false;
                printf("Diferencia en valor %d: paralelo %u, secuencial %u\n", i, h_histogram[i], h_histogram_seq[i]);
            }
        }

        if (!correct) {
            printf("Los histogramas no coinciden en la ejecucion %d\n", run + 1);
        } else {
            printf("Los histogramas coinciden en la ejecucion %d\n", run + 1);
        }
    }

    double parallel_mean = media(parallel_times);
    double parallel_stddev = desvio(parallel_times, parallel_mean);

    double sequential_mean = media(sequential_times);
    double sequential_stddev = desvio(sequential_times, sequential_mean);

    printf("Tiempo de ejecucion en paralelo: media = %f ms, desvio = %f ms\n", parallel_mean * 1000, parallel_stddev * 1000);
    printf("Tiempo de ejecucion secuencial: media = %f ms, desvio = %f ms\n", sequential_mean * 1000, sequential_stddev * 1000);

    // Liberamos memoria
    free(h_image);
    free(h_histogram_matrix);
    CUDA_CHK(cudaFree(d_image));
    CUDA_CHK(cudaFree(d_histogram_matrix));
    CUDA_CHK(cudaFree(d_global_histogram));

    return 0;
}
