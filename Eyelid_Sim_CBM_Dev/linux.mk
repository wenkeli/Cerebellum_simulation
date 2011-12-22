#eyelid conditioning simulation

NAME = cbm_eyelid

CC = icpc
NVCC = nvcc

DEFINES = -DINTELCC

CFLAGS = $(DEFINES) -O3

RM = rm

INTELLIBPATH ='/opt/intel/Compiler/11.0/072/lib/intel64/'
INTELLIBS = -lirc -lcxaguard -limf

CUDAINCPATH = '/opt/cuda/include/'

CBMCOREINCPATH = '/home/consciousness/work/projects/CBM_CORE_LIB/includes/'
CBMCORELIBPATHR = /home/consciousness/work/projects/CBM_CORE_LIB/lib/
CBMCORELIBPATH = '$(CBMCORELIBPATHR)'
CBMCORELIBS = -lcbm_core

INCPATH = ./includes

SRCPATH = ./src

OUTPATH = ./output

MAININC= $(INCPATH)/main.h

MAINSRC = $(SRCPATH)/main.cpp

MAINOBJ = $(OUTPATH)/main.obj

mainapp: main
	-$(CC) $(CFLAGS) $(MAINOBJ) -o $(OUTPATH)/$(NAME) -L $(CBMCORELIBPATH) $(CBMCORELIBS) -Xlinker -R $(CBMCORELIBPATHR)
	
main: $(MAININC) $(MAINSRC)
	-$(CC) $(CFLAGS) -I $(CBMCOREINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/main.cpp -o $(OUTPATH)/main.obj
	
clean:
	-$(RM) $(OUTPATH)/$(NAME)
	-$(RM) $(MAINOBJ)
	
run: $(OUTPATH)/$(NAME)
	-$(OUTPATH)/$(NAME)
	
	