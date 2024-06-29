#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include "radixSort.h"
#include "cuda.h"
#include "cuda_runtime.h"

using namespace std;
using namespace cimg_library;

void filtro_mediana_gpu(float *img_in, float *img_out, int width, int height, int W);
void filtro_mediana_cpu(float *img_in, float *img_out, int width, int height, int W);
int testSplitCPU() {
    float data[] = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
    int n = 1;  // Bit a analizar
    int numElements = sizeof(data) / sizeof(data[0]);
    float output[numElements];

    splitCPU(data, output, n, numElements);

    std::cout << "Array after split on bit " << n << ":\n";
    for (int i = 0; i < numElements; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << "\n";

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
	int a = testSplitCPU();
	return 0;
}