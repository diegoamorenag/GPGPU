#include "radixSort.h"

void exclusiveScanCPU(const float* input, float* output, int numElements) {
    if (numElements > 0) {
        output[0] = 0; // First element is always 0 in an exclusive scan
        for (int i = 1; i < numElements; ++i) {
            output[i] = output[i - 1] + input[i - 1];
        }
    }
}

void exclusiveScanGPU(const float* input, float* output, int numElements) {
    
}

void radixSortCPU(const float* input, float* output, int numElements) {
    
}

void radixSortGPU(const float* input, float* output, int numElements) {
    
}