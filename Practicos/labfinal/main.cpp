#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include "radixSort.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "CImg.h"
#include <bitset>

using namespace std;
using namespace cimg_library;

void filtro_mediana_gpu(float *img_in, float *img_out, int width, int height, int W);
void filtro_mediana_cpu(float *img_in, float *img_out, int width, int height, int W);

void printFloatInBinary(float value)
{
	unsigned int bits;
	memcpy(&bits, &value, sizeof(value));
	std::bitset<32> binary(bits);
	printf("Binary: %s\n", binary.to_string().c_str());
}

int testSplitCPU()
{
	float data[] = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
	int n = 1; // Bit a analizar
	int numElements = sizeof(data) / sizeof(data[0]);
	float output[numElements];

	splitCPU(data, output, n, numElements);

	printf("Array after split on bit %d:\n", n);
	for (int i = 0; i < numElements; i++)
	{
		float value = output[i];
		printf("%f ", value);
		printFloatInBinary(value);
	}
	printf("\n");

	return 0;
}

int main(int argc, char **argv)
{
	const char *path;
	std::string resultsPathCPU;
	std::string resultsPathGPU;

	if (argc < 2)
	{
		printf("Debe ingresar el nombre del archivo\n");
		return 0;
	}
	else
	{
		path = argv[argc - 1];
		resultsPathCPU = "results/" + std::string(argv[argc - 1]) + "output_cpu.ppm";
		resultsPathGPU = "results/" + std::string(argv[argc - 1]) + "output_gpu.ppm";
	}

	CImg<float> image(path);
	CImg<float> image_out(image.width(), image.height(), 1, 1, 0);

	float *img_matrix = image.data();
	float *img_out_matrix = image_out.data();
	float elapsed = 0;

	filtro_mediana_cpu(img_matrix, img_out_matrix, image.width(), image.height(), 3);
	image_out.save(resultsPathCPU.c_str());

	filtro_mediana_gpu(img_matrix, img_out_matrix, image.width(), image.height(), 3);
	image_out.save(resultsPathGPU.c_str());
	testSplitCPU();
	return 0;
}