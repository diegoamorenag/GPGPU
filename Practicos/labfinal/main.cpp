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

    for (int windowSize=3; windowSize<=11; windowSize+2) 
    {
        // Crear la imagen de salida para CPU
        CImg<float> image_out_cpu(width, height, 1, 1, 0);
        float *img_matrix = image.data();
        float *img_out_matrix_cpu = image_out_cpu.data();

        // Aplicar el filtro de mediana en CPU
        filtro_mediana_cpu(img_matrix, img_out_matrix_cpu, width, height, W);

        // Generar el path para el archivo de salida CPU
        std::string resultsPathCPU = "results/" + std::to_string(W) + "/" + filename + "_output_cpu.ppm";
        std::filesystem::create_directories("results/" + std::to_string(W)); // Crear directorio si no existe
        image_out_cpu.save(resultsPathCPU.c_str());

        // Crear la imagen de salida para GPU
        CImg<float> image_out_gpu(width, height, 1, 1, 0);
        float *img_out_matrix_gpu = image_out_gpu.data();

        // Aplicar el filtro de mediana en GPU
        filtro_mediana_gpu(img_matrix, img_out_matrix_gpu, width, height, W);

        // Generar el path para el archivo de salida GPU
        std::string resultsPathGPU = "results/" + std::to_string(W) + "/" + filename + "_output_gpu.ppm";
        image_out_gpu.save(resultsPathGPU.c_str());
    }
	return 0;
}
