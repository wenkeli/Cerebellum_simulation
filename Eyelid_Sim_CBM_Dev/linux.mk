#eyelid conditioning simulation

NAME = cbm_eyelid

CC = icpc
NVCC = nvcc

DEFINES = -DINTELCC

CFLAGS = $(DEFINES) -O3

RM = rm
MOC = moc
UIC = uic

INTELLIBPATH ='/opt/intel/Compiler/11.0/072/lib/intel64/'
INTELLIBS = -lirc -lcxaguard -limf

QTINCPATH = '/usr/include/qt4/'
QTLIBPATH = '/usr/lib/qt4/'
QTLIBS = -lQtGui -lQtCore

CUDAINCPATH = '/opt/cuda/include/'

CBMCOREINCPATH = '/home/consciousness/work/projects/CBM_CORE_LIB/includes/'
CBMCORELIBPATHR = /home/consciousness/work/projects/CBM_CORE_LIB/lib/
CBMCORELIBPATH = '$(CBMCORELIBPATHR)'
CBMCORELIB = cbm_core

CBMVISUALINCPATH = '/home/consciousness/work/projects/CBM_VISUAL_LIB/includes/'
CBMVISUALLIBPATHR = /home/consciousness/work/projects/CBM_VISUAL_LIB/lib/
CBMVISUALLIBPATH = '$(CBMVISUALLIBPATHR)'
CBMVISUALLIB = cbm_visual

EXTINCPATH = -I $(QTINCPATH) -I $(CUDAINCPATH) -I $(CBMCOREINCPATH) -I $(CBMVISUALINCPATH)

LIBS = $(CBMCORELIBS) $(CBMVISUALLIBS)

INCPATH = ./includes
GUIIP = $(INCPATH)/gui
GUIMOCIP = $(GUIIP)/moc
GUIUICIP = $(GUIIP)/uic
GUIUIIP = $(GUIIP)/ui

SRCPATH = ./src
GUISP = $(SRCPATH)/gui

OUTPATH = ./output

UIS = $(GUIUIIP)/mainw.ui
MOCINC = $(GUIIP)/mainw.h
UICOUT = $(GUIUICIP)/ui_mainw.h
MOCOUT = $(GUIMOCIP)/moc_mainw.h
GUIINC = $(MOCINC) $(MOCOUT) $(UICOUT)

MAININC= $(INCPATH)/main.h

INC = $(GUIINC) $(MAININC)

GUISRC = $(GUISP)/mainw.cpp
MAINSRC = $(SRCPATH)/main.cpp

SRC = $(GUISRC) $(MAINSRC)

GUIOBJ = $(OUTPATH)/mainw.obj
MAINOBJ = $(OUTPATH)/main.obj

OBJ = $(GUIOBJ) $(MAINOBJ)

mainapp: main gui
	-$(CC) $(CFLAGS) $(OBJ) -o $(OUTPATH)/$(NAME) \
	-L$(QTLIBPATH) $(QTLIBS) -L$(INTELLIBPATH) $(INTELLIBS) \
	-Xlinker -rpath=$(CBMCORELIBPATHR) -Xlinker -rpath=$(CBMVISUALLIBPATHR) \
	-Xlinker --library-path=$(CBMCORELIBPATHR) -Xlinker --library-path=$(CBMVISUALLIBPATHR) \
	-Xlinker --library=$(CBMCORELIB) -Xlinker --library=$(CBMVISUALLIB)
	
main: $(MAININC) $(MAINSRC)
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(SRCPATH)/main.cpp -o $(OUTPATH)/main.obj
	
gui: $(GUIINC) $(GUISRC)
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(GUISP)/mainw.cpp -o $(OUTPATH)/mainw.obj

guiinc: uics mocs
	
mocs: $(MOCINC)
	-$(MOC) $(GUIIP)/mainw.h -o $(GUIMOCIP)/moc_mainw.h
	
uics: $(UIS)
	-$(UIC) $(GUIUIIP)/mainw.ui -o $(GUIUICIP)/ui_mainw.h
	
clean:
	-$(RM) $(OUTPATH)/$(NAME)
	-$(RM) $(OBJ)
	
run: $(OUTPATH)/$(NAME)
	-$(OUTPATH)/$(NAME)
	