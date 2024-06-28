#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "CImg.h"

using namespace cimg_library;

void filtro_mediana_gpu(float * img_in, float * img_out, int width, int height, int W){

}
void filtro_mediana_cpu(float * img_in, float * img_out, int width, int height, int W){

}
    
int main(int argc, char** argv){
	const char * path;

	if (argc < 2) printf("Debe ingresar el nombre del archivo\n");
	else
		path = argv[argc-1];

	CImg<float> image(path);
	CImg<float> image_out(image.width(), image.height(),1,1,0);

	float *img_matrix = image.data();
   float *img_out_matrix = image_out.data();

	float elapsed = 0;

	filtro_mediana_cpu(img_matrix, img_out_matrix, image.width(), image.height(), 3);
	image_out.save("output_cpu.ppm");

	filtro_mediana_gpu(img_matrix, img_out_matrix, image.width(), image.height(), 3);
	image_out.save("output_gpu.ppm");
   	
   return 0;
}

