all:
	nvcc -arch=sm_60 -Xptxas main.cpp blur.cu -o blur -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -lnvToolsExt
