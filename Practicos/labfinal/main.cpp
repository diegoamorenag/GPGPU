#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include "radixSort.h"
#include <bitset>
#include "cuda.h"
#include "cuda_runtime.h"
#include "CImg.h"

using namespace std;
using namespace cimg_library;

void filtro_mediana_gpu(float *img_in, float *img_out, int width, int height, int W);
void filtro_mediana_cpu(float *img_in, float *img_out, int width, int height, int W);

void printFloatInBinary(float value)
{
	unsigned int bits;
	memcpy(&bits, &value, sizeof(value));
	std::bitset<32> binary(bits);
	printf(" Binario: %s\n", binary.to_string().c_str());
}

int testSplitCPU()
{
	byte data[] = {3.42, 1.47, 2.42, 6.84, 5.88, 3.12, 5.1};
	byte output[11]= {0,0,0,0,0,0,0,0,0,0,0};
	int numElements = sizeof(data) / sizeof(data[0]);
	for (int i = 7; i >=0 ; i--)
	{
		printf("bit: %d:------------------\n\n",i );
		for (int bitAOrdenar = 0; bitAOrdenar < numElements; bitAOrdenar++)
		{
			float value = output[bitAOrdenar];
			printf("float: %f ", value);
			printFloatInBinary(value);
		}
		splitCPU(data, output, i, numElements);
		for (int j = 0; j < numElements; j++)
		{
			data[j] = output[j];
		}
	}
	return 0;
}

int main(int argc, char **argv)
{
	//const char *path;
	//std::string resultsPathCPU;
	//std::string resultsPathGPU;
//
	//if (argc < 2)
	//{
	//	printf("Debe ingresar el nombre del archivo\n");
	//	return 0;
	//}
	//else
	//{
	//	path = argv[argc - 1];
	//	resultsPathCPU = "results/" + std::string(argv[argc - 1]) + "output_cpu.ppm";
	//	resultsPathGPU = "results/" + std::string(argv[argc - 1]) + "output_gpu.ppm";
	//}
//
	//CImg<float> image(path);
	//CImg<float> image_out(image.width(), image.height(), 1, 1, 0);
//
	//float *img_matrix = image.data();
	//float *img_out_matrix = image_out.data();
	//float elapsed = 0;
//
	//filtro_mediana_cpu(img_matrix, img_out_matrix, image.width(), image.height(), 3);
	//image_out.save(resultsPathCPU.c_str());
//
	//filtro_mediana_gpu(img_matrix, img_out_matrix, image.width(), image.height(), 3);
	//image_out.save(resultsPathGPU.c_str());
	testSplitCPU();
	return 0;
}