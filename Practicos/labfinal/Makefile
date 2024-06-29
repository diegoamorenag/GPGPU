OUTDIR = output

all: $(OUTDIR) blur

blur: $(OUTDIR)/main.o $(OUTDIR)/blur.o
	nvcc -arch=sm_60 -dlink $(OUTDIR)/main.o blur.o -o $(OUTDIR)/dlink.o -O3
	nvcc -arch=sm_60 -o blur $(OUTDIR)/main.o blur.o $(OUTDIR)/dlink.o -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -lnvToolsExt

$(OUTDIR)/main.o: main.cpp
	nvcc -arch=sm_60 -c main.cpp -o $(OUTDIR)/main.o -O3

$(OUTDIR)/blur.o: blur.cu
	nvcc -arch=sm_60 -c blur.cu -o blur.o -O3

$(OUTDIR)/radixSort.o: radixSort.cpp
    nvcc -arch=sm_60 -c radixSort.cpp -o $(OUTDIR)/radixSort.o -O3

clean:
	rm -f blur $(OUTDIR)/* 

#nvcc -arch=sm_60 -Xptxas main.cpp blur.cu -o blur -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -lnvToolsExt