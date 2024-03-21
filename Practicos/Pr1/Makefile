CXX = g++
CXXFLAGS = -std=c++17
OUTPUT_DIR_1 = Ej1/output
OUTPUT_DIR_2 = Ej2/output

all: ej1compile ej2compile ej1run ej2run

ej1compile: $(OUTPUT_DIR_1)/1a $(OUTPUT_DIR_1)/1bSinPrefetch $(OUTPUT_DIR_1)/1bConPrefetch $(OUTPUT_DIR_1)/1c

$(OUTPUT_DIR_1)/1a: Ej1/1a.cpp auxFunctions.o
	$(CXX) $(CXXFLAGS) -O0 Ej1/1a.cpp auxFunctions.o -o $@

$(OUTPUT_DIR_1)/1bSinPrefetch: Ej1/1b.cpp auxFunctions.o
	$(CXX) -O0 Ej1/1b.cpp auxFunctions.o -o $@

$(OUTPUT_DIR_1)/1bConPrefetch: Ej1/1b.cpp auxFunctions.o
	$(CXX) -fprefetch-loop-arrays Ej1/1b.cpp auxFunctions.o -o $@

$(OUTPUT_DIR_1)/1c: Ej1/1c.cpp auxFunctions.o
	$(CXX) -O0 Ej1/1c.cpp auxFunctions.o -o $@

ej1run: ej1arun ej1brun ej1crun

ej1arun:
	echo "Ejercicio 1-A\n"
	$(OUTPUT_DIR_1)/1a

ej1brun:
	echo "Ejercicio 1-B\n"
	$(OUTPUT_DIR_1)/1bSinPrefetch
	$(OUTPUT_DIR_1)/1bConPrefetch

ej1crun:
	echo "Ejercicio 1-C\n"
	$(OUTPUT_DIR_1)/1c

ej2compile: $(OUTPUT_DIR_2)/2a $(OUTPUT_DIR_2)/2b $(OUTPUT_DIR_2)/2c

$(OUTPUT_DIR_2)/2a: Ej2/2a.cpp auxFunctions.o
	$(CXX) Ej2/2a.cpp auxFunctions.o -o $@

$(OUTPUT_DIR_2)/2b: Ej2/2b.cpp auxFunctions.o
	$(CXX) Ej2/2b.cpp auxFunctions.o -o $@

$(OUTPUT_DIR_2)/2c: Ej2/2c.cpp auxFunctions.o
	$(CXX) Ej2/2c.cpp auxFunctions.o -o $@

ej2run: ej2arun ej2brun ej2crun

ej2arun:
	echo "Ejercicio 2-A\n"
	$(OUTPUT_DIR_2)/2a

ej2brun:
	echo "Ejercicio 2-B\n"
	$(OUTPUT_DIR_2)/2b

ej2crun:
	echo "Ejercicio 2-C\n"
	$(OUTPUT_DIR_2)/2c

auxFunctions.o: auxFunctions.cpp auxFunctions.h
	$(CXX) $(CXXFLAGS) -c auxFunctions.cpp

2a.o: 2a.cpp
	$(CXX) $(CXXFLAGS) -c 2a.cpp

2b.o: 2b.cpp
	$(CXX) $(CXXFLAGS) -c 2b.cpp

2c.o: 2c.cpp
	$(CXX) $(CXXFLAGS) -c 2c.cpp

clean:
	rm -f *.o $(OUTPUT_DIR_1)/* $(OUTPUT_DIR_2)/*
