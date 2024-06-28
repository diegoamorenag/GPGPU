all: blur

blur: main.o blur.o
	nvcc -arch=sm_60 -Xptxas main.o blur.o -o blur -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -lnvToolsExt

main.o: main.cpp
	nvcc -arch=sm_60 -Xptxas -dc main.cpp -o main.o -O3

blur.o: blur.cu
	nvcc -arch=sm_60 -Xptxas -dc blur.cu -o blur.o -O3

clean:
	rm -f blur main.o blur.o
#nvcc -arch=sm_60 -Xptxas main.cpp blur.cu -o blur -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -lnvToolsExt