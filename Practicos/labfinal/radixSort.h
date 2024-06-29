// exclusiveScan.h
#ifndef RADIX_SPRT_H
#define RADIX_SPRT_H

void exclusiveScanCPU(const float* input, float* output, int numElements);

void exclusiveScanGPU(const float* input, float* output, int numElements);

void radixSortCPU(const float* input, float* output, int numElements);

void radixSortGPU(const float* input, float* output, int numElements);

#endif // RADIX_SPRT_H