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

void exclusiveScanCPU(const byte *input, byte *output, int numElements)
{
    if (numElements > 0)
    {
        output[0] = 0;
        for (int i = 1; i < numElements; ++i)
        {
            output[i] = output[i - 1] + input[i - 1];
        }
    }
}

void splitCPU(const byte *input, byte *output, int n, int numElements)
{
    std::vector<byte> e(numElements);
    std::vector<byte> scanResults(numElements);

    for (int i = 0; i < numElements; i++)
    {
        byte value = input[i];
        memcpy(&bits, &value, sizeof(value));
        std::bitset<32> binary(bits);
        printf(bits);;
        byte intValue = static_cast<int> (input[i]);
        printf(~(intValue >> n) & 1);
        e[i] = static_cast<byte>(~(intValue >> n) & 1);
        unsigned byte bits;
        memcpy(&bits, &value, sizeof(value));
        std::bitset<32> binary(bits);
    }
    // Perform exclusive scan
    exclusiveScanCPU(e.data(), scanResults.data(), numElements);
    // Debug: Print results of exclusive scan
    printf("Results of exclusive scan:\n");
    for (int i = 0; i < numElements; i++)
    {
        printf("ScanResults[%d] = %f\n", i, scanResults[i]);
    }
    // Calculate total number of false (0 bit values)
    int totalFalses = static_cast<int>(scanResults[numElements - 1] + (((static_cast<int>(input[numElements - 1]) >> n) & 1) == 0 ? 1 : 0));
    printf("Total falses: %d\n", totalFalses);
    // Assign outputs based on calculated indices
    for (int i = 0; i < numElements; i++)
    {
        int intValue = static_cast<int>(input[i]);
        if ((intValue >> n) & 1)
        {
            output[int(i - scanResults[i] + totalFalses)] = input[i];
            printf("Output[%d] = Input[%d] = %f (1-bit)\n", int(i - scanResults[i] + totalFalses), i, input[i]);
        }
        else
        {
            output[int(scanResults[i])] = input[i];
            printf("Output[%d] = Input[%d] = %f (0-bit)\n", int(scanResults[i]), i, input[i]);
        }
    }
}

void radixSortCPU(const 
float *input, float *output, int numElements)
{
}

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