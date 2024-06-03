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

__global__ void histogram_kernel(int *image, unsigned int *histogram_matrix, int width, int height) {
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

__global__ void reduction_kernel(unsigned int *histogram_matrix, unsigned int *global_histogram, int num_blocks) {
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
        global_histogram[col_id] = intermedio[0];
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
        histogram_kernel<<<gridSize, blockSize>>>(d_image, d_histogram_matrix, WIDTH, HEIGHT);
        CUDA_CHK(cudaDeviceSynchronize());
        
        reduction_kernel<<<HISTOGRAM_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(unsigned int)>>>(d_histogram_matrix, d_global_histogram, num_blocks);
        CUDA_CHK(cudaDeviceSynchronize());
        
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

    double media_calculo_paralelo = media(parallel_times);
    double desvio_en_paralelo = desvio(parallel_times, media_calculo_paralelo);

    double media_secuencial = media(sequential_times);
    double desvio_secuencial = desvio(sequential_times, media_secuencial);

    printf("Tiempo de ejecucion en paralelo: media = %f ms, desvio = %f ms\n", media_calculo_paralelo * 1000, desvio_en_paralelo * 1000);
    printf("Tiempo de ejecucion secuencial: media = %f ms, desvio = %f ms\n", media_secuencial * 1000, desvio_secuencial * 1000);

    // Liberamos memoria
    free(h_image);
    free(h_histogram_matrix);
    CUDA_CHK(cudaFree(d_image));
    CUDA_CHK(cudaFree(d_histogram_matrix));
    CUDA_CHK(cudaFree(d_global_histogram));

    return 0;
}
