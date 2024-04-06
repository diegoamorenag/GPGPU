#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void read_file(const char*, int*);
int get_text_length(const char * fname);

#define A 15
#define B 27
#define M 256
#define A_MMI_M -17

#define N 1024


__device__ int modulo(int a, int b){
	int r = a % b;
	r = (r < 0) ? r + b : r;
	return r;
}

__global__ void count_occurrences_kernel(int *d_text, int length, unsigned int *d_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for (int i = idx; i < length; i += step) {
        atomicAdd(&d_count[d_text[i]], 1);
    }
}

int main(int argc, char *argv[])
{
    int *h_message, *d_message;
    unsigned int *d_count, h_count[256];
    unsigned int size;

    const char * fname;

	if (argc < 2) printf("Debe ingresar el nombre del archivo\n");
	else
		fname = argv[1];

    int length = get_text_length(fname);
    size = length * sizeof(int);

    // Reserva de memoria para el mensaje en la CPU
    h_message = (int *)malloc(size);

    // Lectura del archivo
    read_file(fname, h_message);

    // Reserva de memoria para el mensaje y el conteo en la GPU
    CUDA_CHK(cudaMalloc((void**)&d_message, size));
    CUDA_CHK(cudaMalloc((void**)&d_count, 256 * sizeof(unsigned int)));
    CUDA_CHK(cudaMemset(d_count, 0, 256 * sizeof(unsigned int)));

    // Copia del mensaje a la GPU
    CUDA_CHK(cudaMemcpy(d_message, h_message, size, cudaMemcpyHostToDevice));

    // Lanzamiento del kernel para contar las ocurrencias de cada carácter
    int threadsPerBlock = 256;
    int numberOfBlocks = (length + threadsPerBlock - 1) / threadsPerBlock;
    count_occurrences_kernel<<<numberOfBlocks, threadsPerBlock>>>(d_message, length, d_count);
    CUDA_CHK(cudaDeviceSynchronize());

    // Copia del conteo de vuelta a la CPU
    CUDA_CHK(cudaMemcpy(h_count, d_count, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // Mostrar el conteo de ocurrencias de cada carácter
    for (int i = 0; i < 256; i++) {
        if (h_count[i] > 0) {
            printf("Carácter '%c' (ASCII %d): %d veces\n", i, i, h_count[i]);
        }
    }

    // Liberación de memoria
    free(h_message);
    CUDA_CHK(cudaFree(d_message));
    CUDA_CHK(cudaFree(d_count));

    return 0;
}
	
int get_text_length(const char * fname)
{
	FILE *f = NULL;
	f = fopen(fname, "r"); //read and binary flags

	size_t pos = ftell(f);    
	fseek(f, 0, SEEK_END);    
	size_t length = ftell(f); 
	fseek(f, pos, SEEK_SET);  

	fclose(f);

	return length;
}

void read_file(const char * fname, int* input)
{
	// printf("leyendo archivo %s\n", fname );

	FILE *f = NULL;
	f = fopen(fname, "r"); //read and binary flags
	if (f == NULL){
		fprintf(stderr, "Error: Could not find %s file \n", fname);
		exit(1);
	}

	//fread(input, 1, N, f);
	int c; 
	while ((c = getc(f)) != EOF) {
		*(input++) = c;
	}

	fclose(f);
}
