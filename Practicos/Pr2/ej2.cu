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

#define N 512


__device__ int modulo(int a, int b){
	int r = a % b;
	r = (r < 0) ? r + b : r;
	return r;
}

__global__ void contar_caracteres(unsigned char *text, int length, int *occurrences) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < length) {
        unsigned char character = text[idx];
        atomicAdd(&occurrences[character], 1);
    }
}

int main(int argc, char *argv[])
{
	int *text;
	unsigned int size;

	const char * fname;

	if (argc < 2) printf("Debe ingresar el nombre del archivo\n");
	else
		fname = argv[1];

	int length = get_text_length(fname);

	size = length * sizeof(int);

	// reservar memoria para el texto
	text = (int *)malloc(size);

	// leo el archivo de la entrada
	read_file(fname, text);

	unsigned char *d_text;
	int *d_occurrences;
	int h_occurrences[256] = {0};	//inicializado en ceros

	CUDA_CHK(cudaMalloc((void**)&d_text, length));
	CUDA_CHK(cudaMalloc((void**)&d_occurrences, 256 * sizeof(int)));
	CUDA_CHK(cudaMemcpy(d_text, text, length, cudaMemcpyHostToDevice));
	CUDA_CHK(cudaMemset(d_occurrences, 0, 256 * sizeof(int)));

	int threadsPerBlock = N;
	int blocksPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;
	contar_caracteres<<<blocksPerGrid, threadsPerBlock>>>(d_text, length, d_occurrences);

	CUDA_CHK(cudaMemcpy(h_occurrences, d_occurrences, 256 * sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = 0; i < 256; i++) {
		if (h_occurrences[i] > 0) {
			printf("Caracter %c -- Ocurrencias: %d\n", i, h_occurrences[i]);
		}
	}

	// Escribir el mensaje desencriptado en texto.txt
	FILE *f_out = fopen("ocurrencias.txt", "w");
	if (f_out == NULL) {
	    fprintf(stderr, "Error: No se pudo abrir ocurrencias.txt para escritura\n");
	    exit(1);
	}

	for (int i = 0; i < 256; i++) {
	    fprintf(f_out, "%c", (char)h_occurrences[i]);
	}

	fclose(f_out);

	//liberar memoria de gpu
	CUDA_CHK(cudaFree(d_text));
	CUDA_CHK(cudaFree(d_occurrences));

	//liberar memoria de cpu
	free(text);

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
