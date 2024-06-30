#include "radixSort.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <bitset>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/time.h>

void exclusiveScanCPU(const float *input, float *output, int numElements)
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

void splitCPU(const float *input, float *output, int n, int numElements)
{
    std::vector<float> e(numElements);
    std::vector<float> scanResults(numElements);

    printf("Input values and their binary representations:\n");
    for (int i = 0; i < numElements; i++)
    {
        float value = input[i];
        int intValue = static_cast<int>(input[i]);
        e[i] = static_cast<float>(~(intValue >> n) & 1);
        unsigned int bits;
        memcpy(&bits, &value, sizeof(value));
        std::bitset<32> binary(bits);
        printf("Binario: %s\n", binary.to_string().c_str());
        printf("Input[%d] = %f, IntValue = %d, Bit[%d] = %f\n", i, input[i], intValue, n, e[i]);
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

void radixSortCPU(const float *input, float *output, int numElements)
{
}

void exclusiveScanGPU(const float *input, float *output, int numElements)
{
}

void splitGPU(const int *input, int *output, int n)
{
}

void radixSortGPU(const float *input, float *output, int numElements)
{
}