#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include "radixSort.h"
#include <bitset>
#include "cuda.h"
#include "cuda_runtime.h"
#include "CImg.h"
#include <vector>
#include <iostream>
#include "radixSort.h"


void exclusiveScanGPU(const float *input, float *output, int numElements)
{
    float *d_input = nullptr;
    float *d_output = nullptr;
    cudaMalloc(&d_input, numElements * sizeof(int));
    cudaMalloc(&d_output, numElements * sizeof(int));
    cudaMemcpy(d_input, input.data(), numElements * sizeof(int), cudaMemcpyHostToDevice);

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_input, d_output, n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_input, d_output, n);

    cudaMemcpy(output.data(), d_output, numElements * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp_storage);
}

void splitGPU(const int *input, int *output, int n)
{
}

void radixSortGPU(const float *input, float *output, int numElements)
{
}