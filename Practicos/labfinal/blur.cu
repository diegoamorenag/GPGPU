#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "nvToolsExt.h"


using namespace std;

__global__ void filtro_mediana_kernel(float* d_input, float* d_output, int width, int height, float W){

}

void filtro_mediana_gpu(float * img_in, float * img_out, int width, int height, int W){

}

void filtro_mediana_cpu(float * img_in, float * img_out, int width, int height, int W){

}
