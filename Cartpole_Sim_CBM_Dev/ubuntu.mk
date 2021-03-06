#cartpole simulation

NAME = cbm_cartpole

CC = g++
NVCC = nvcc

DEFINES = 

CFLAGS = $(DEFINES) -O3

RM = rm

INTELLIBPATH =
INTELLIBS = 

CUDAINCPATH = '/usr/local/cuda/include/'

CBMCOREINCPATH = '/home/mhauskn/projects/CBM_CORE_LIB_MGPU_DEV/includes/'
CBMCORELIBPATHR = /home/mhauskn/projects/CBM_CORE_LIB_MGPU_DEV/lib/
CBMCORELIBPATH = '$(CBMCORELIBPATHR)'
CBMCORELIBS = -lcbm_core

VIZPATH = /home/mhauskn/projects/cerebellumViz
VIZINCPATH = $(VIZPATH)/inc
VIZLIBPATH = $(VIZPATH)/lib
VIZLIBS = -lcbmViz_x86_64

GLUTLIBS = -lglut -lGL -lGLU -lX11 -lXmu -lGLEW

INCPATH = ./includes

SRCPATH = ./src

OUTPATH = ./output

MAININC= $(INCPATH)/main.h

MAINSRC = $(SRCPATH)/main.cpp

MAINOBJ = $(OUTPATH)/main.obj

mainapp: main
	-$(CC) $(CFLAGS) $(MAINOBJ) -o $(OUTPATH)/$(NAME) -L $(CBMCORELIBPATH) -L $(VIZLIBPATH) $(VIZLIBS) $(CBMCORELIBS) $(GLUTLIBS) -Xlinker -R $(CBMCORELIBPATHR)

main: $(MAININC) $(MAINSRC)
	-$(CC) $(CFLAGS) -I $(CBMCOREINCPATH) -I $(CUDAINCPATH) -I $(VIZINCPATH) -c $(SRCPATH)/main.cpp -o $(OUTPATH)/main.obj

clean:
	-$(RM) $(OUTPATH)/$(NAME)
	-$(RM) $(MAINOBJ)

run: $(OUTPATH)/$(NAME)
	-$(OUTPATH)/$(NAME)
