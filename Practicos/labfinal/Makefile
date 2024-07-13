# Compiler settings
CXX = g++
NVCC = nvcc
CXXFLAGS = -Wall -std=c++11 -O2
NVCCFLAGS = -O2

# Directories
CPU_DIR = CPU
GPU_DIR = GPU

# Targets
TARGETS = $(CPU_DIR)/median_filter \
          $(GPU_DIR)/baseline/baseline \
          $(GPU_DIR)/memoria_compartida/memoria_compartida \
          $(GPU_DIR)/network/network \
          $(GPU_DIR)/quickselect/quickselect \
          $(GPU_DIR)/radixCub/radixCub \
          $(GPU_DIR)/radixCurso/radixCurso \
          $(GPU_DIR)/texturas/texturas

# Default target
all: $(TARGETS)

# CPU target
$(CPU_DIR)/median_filter: $(CPU_DIR)/median_filter.cpp
	$(CXX) $(CXXFLAGS) -o $@ $

# GPU targets
$(GPU_DIR)/%/% : $(GPU_DIR)/%/%.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $

# Clean
clean:
	rm -f $(TARGETS)

# Run targets
run_cpu: $(CPU_DIR)/median_filter
	./$(CPU_DIR)/median_filter input.pgm output.pgm 3

run_gpu_%: $(GPU_DIR)/%/%
	./$(GPU_DIR)/$*/$* input.pgm output.pgm 3

# Phony targets
.PHONY: all clean run_cpu $(addprefix run_gpu_,baseline memoria_compartida network quickselect radixCub radixCurso texturas)