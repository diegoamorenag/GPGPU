# Routes
CUB = /clusteruy/home/gpgpu10/cub/cub
MKLROOT = /home/gpgpu/software/mkl
CUDA_INSTALL_PATH = /usr/local/cuda-12.1

# Compiler settings
CXX = g++
NVCC = nvcc
CXXFLAGS = -Wall -std=c++11 -O2
NVCCFLAGS = -std=c++11 -O2 -arch=sm_60 -I$(CUDA_INSTALL_PATH)/include -I$(CUB)

# Directories
CPU_DIR = CPU
GPU_DIR = GPU

# Targets
TARGETS = $(CPU_DIR)/median_filter \
          $(GPU_DIR)/baseline/baseline \
		  $(GPU_DIR)/counting/counting \
          $(GPU_DIR)/memoria_compartida/memoria_compartida \
          $(GPU_DIR)/network/network \
          $(GPU_DIR)/quickselect/quickselect \
          $(GPU_DIR)/radix_filter/radix_filter \
          $(GPU_DIR)/radixCub/radixCub \
          $(GPU_DIR)/radixCurso/radixCurso \
		  $(GPU_DIR)/shell/shell \
		  $(GPU_DIR)/texturas/texturas \
		  $(GPU_DIR)/thrust_filter/thrust_filter \

# Default target
all: $(TARGETS)

# CPU target
CPU/median_filter: CPU/median_filter.cpp
	g++ -Wall -std=c++11 -O2 -o CPU/median_filter CPU/median_filter.cpp

# GPU targets
GPU/baseline/baseline: GPU/baseline/baseline.cu
	/usr/local/cuda-12.1/bin/nvcc -std=c++11 -O2 -arch=sm_60 -I/usr/local/cuda-12.1/include -I/clusteruy/home/gpgpu10/cub/cub -o GPU/baseline/baseline GPU/baseline/baseline.cu

GPU/counting/counting: GPU/counting/counting.cu	
	/usr/local/cuda-12.1/bin/nvcc -std=c++11 -O2 -arch=sm_60 -I/usr/local/cuda-12.1/include -I/clusteruy/home/gpgpu10/cub/cub -o GPU/counting/counting GPU/counting/counting.cu

GPU/memoria_compartida/memoria_compartida: GPU/memoria_compartida/memoria_compartida.cu
	/usr/local/cuda-12.1/bin/nvcc -std=c++11 -O2 -arch=sm_60 -I/usr/local/cuda-12.1/include -I/clusteruy/home/gpgpu10/cub/cub -o GPU/memoria_compartida/memoria_compartida GPU/memoria_compartida/memoria_compartida.cu

GPU/network/network: GPU/network/network.cu
	/usr/local/cuda-12.1/bin/nvcc -std=c++11 -O2 -arch=sm_60 -I/usr/local/cuda-12.1/include -I/clusteruy/home/gpgpu10/cub/cub -o GPU/network/network GPU/network/network.cu

GPU/quickselect/quickselect: GPU/quickselect/quickselect.cu
	/usr/local/cuda-12.1/bin/nvcc -std=c++11 -O2 -arch=sm_60 -I/usr/local/cuda-12.1/include -I/clusteruy/home/gpgpu10/cub/cub -o GPU/quickselect/quickselect GPU/quickselect/quickselect.cu

GPU/radixCub/radixCub: GPU/radixCub/radixCub.cu
	/usr/local/cuda-12.1/bin/nvcc -std=c++11 -O2 -arch=sm_60 -I/usr/local/cuda-12.1/include -I/clusteruy/home/gpgpu10/cub/cub -o GPU/radixCub/radixCub GPU/radixCub/radixCub.cu

GPU/radixCurso/radixCurso: GPU/radixCurso/radixCurso.cu
	/usr/local/cuda-12.1/bin/nvcc -std=c++11 -O2 -arch=sm_60 -I/usr/local/cuda-12.1/include -I/clusteruy/home/gpgpu10/cub/cub -o GPU/radixCurso/radixCurso GPU/radixCurso/radixCurso.cu

GPU/radix_filter/radix_filter: GPU/radix_filter/radix_filter.cu
	/usr/local/cuda-12.1/bin/nvcc -std=c++11 -O2 -arch=sm_60 -I/usr/local/cuda-12.1/include -I/clusteruy/home/gpgpu10/cub/cub -o GPU/radix_filter/radix_filter GPU/radix_filter/radix_filter.cu

GPU/shell/shell: GPU/shell/shell.cu
	/usr/local/cuda-12.1/bin/nvcc -std=c++11 -O2 -arch=sm_60 -I/usr/local/cuda-12.1/include -I/clusteruy/home/gpgpu10/cub/cub -o GPU/shell/shell GPU/shell/shell.cu

GPU/texturas/texturas: GPU/texturas/texturas.cu
	/usr/local/cuda-12.1/bin/nvcc -std=c++11 -O2 -arch=sm_60 -I/usr/local/cuda-12.1/include -I/clusteruy/home/gpgpu10/cub/cub -o GPU/texturas/texturas GPU/texturas/texturas.cu

GPU/thrust_filter/thrust_filter: GPU/thrust_filter/thrust_filter.cu
	/usr/local/cuda-12.1/bin/nvcc -o GPU/thrust_filter/thrust_filter GPU/thrust_filter/thrust_filter.cu

# Clean
clean:
	rm -f $(TARGETS)

# Run targets
run_cpu: $(CPU_DIR)/median_filter
	./$(CPU_DIR)/median_filter input.pgm output.pgm 3

run_gpu_%: $(GPU_DIR)/%/%
	./$(GPU_DIR)/$*/$* input.pgm output.pgm 3

.PHONY: run-algorithms
run-algorithms:
	./run_algorithms.sh
