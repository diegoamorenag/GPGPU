#include "radixSort.h"
#include <vector>

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

void splitCPU(const float *input, float *output, int n, int numElements) {
    std::vector<float> e(numElements);

    for (int i = 0; i < numElements; i++) {
        int intValue = static_cast<int>(input[i]);
        e[i] = static_cast<float>(~(intValue >> n) & 1);
    }
    std::vector<float> scanResults(numElements);
    // 2. Llamar a exclusiveScanCPU para realizar la suma prefija exclusiva
    exclusiveScanCPU(e.data(), scanResults.data(), numElements);

    // Determinar el total de falses (total de ceros)
    int totalFalses = static_cast<int>(scanResults[numElements - 1] + (((static_cast<int>(input[numElements - 1]) >> n) & 1) == 0 ? 1 : 0));
    // 3. Calcular los índices de los elementos en el arreglo de salida
    for (int i = 0; i < numElements; i++) {
        int intValue = static_cast<int>(input[i]);
        if ((intValue >> n) & 1) {
            // Índice para los elementos con b = 1
            output[int(i - scanResults[i] + totalFalses)] = input[i];
        } else {
            // Índice para los elementos con b = 0
            output[int(scanResults[i])] = input[i];
        }
    }
}

void radixSortCPU(const float *input, float *output, int numElements)
{
}

void exclusiveScanGPU(const float *input, float *output, int numElements)
{
}

void splitGPU(const int *input,int *output, int n)
{
}

void radixSortGPU(const float *input, float *output, int numElements)
{
}