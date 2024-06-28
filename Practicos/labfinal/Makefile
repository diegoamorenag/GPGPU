all: blur

blur: main.o blur.o
	nvcc -arch=sm_60 -dlink main.o blur.o -o dlink.o -O3
	nvcc -arch=sm_60 -o blur main.o blur.o dlink.o -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -lnvToolsExt

main.o: main.cpp
	nvcc -arch=sm_60 -c main.cpp -o main.o -O3

blur.o: blur.cu
	nvcc -arch=sm_60 -c blur.cu -o blur.o -O3

clean:
	rm -f blur main.o blur.o dlink.o

#nvcc -arch=sm_60 -Xptxas main.cpp blur.cu -o blur -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -lnvToolsExt