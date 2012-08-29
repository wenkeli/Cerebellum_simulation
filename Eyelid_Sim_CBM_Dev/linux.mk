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

CBMCOREINCPATH = '../CBM_CORE_LIB/includes/'
CBMCORELIBPATHR = ../CBM_CORE_LIB/lib/
CBMCORELIBPATH = '$(CBMCORELIBPATHR)'
CBMCORELIB = cbm_core

CBMVISUALINCPATH = '../CBM_VISUAL_LIB/'
CBMVISUALLIBPATHR = ../CBM_VISUAL_LIB/lib/
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

UIS = $(GUIUIIP)/mainw.ui $(GUIUIIP)/testpanel.ui
MOCINC = $(GUIIP)/mainw.h $(GUIIP)/testpanel.h $(GUIIP)/simthread.h
UICOUT = $(GUIUICIP)/ui_mainw.h $(GUIUICIP)/ui_testpanel.h
MOCOUT = $(GUIMOCIP)/moc_mainw.h $(GUIMOCIP)/moc_testpanel.h $(GUIMOCIP)/moc_simthread.h
GUIINC = $(MOCINC) $(MOCOUT) $(UICOUT)

MAININC= $(INCPATH)/main.h $(INCPATH)/ecmanagement.h

INC = $(GUIINC) $(MAININC)

GUISRC = $(GUISP)/mainw.cpp $(GUISP)/testpanel.cpp $(GUISP)/simthread.cpp
MAINSRC = $(SRCPATH)/main.cpp $(SRCPATH)/ecmanagement.cpp

SRC = $(GUISRC) $(MAINSRC)

GUIOBJ = $(OUTPATH)/mainw.obj $(OUTPATH)/testpanel.obj $(OUTPATH)/simthread.obj
MAINOBJ = $(OUTPATH)/main.obj $(OUTPATH)/ecmanagement.obj

OBJ = $(GUIOBJ) $(MAINOBJ)

mainapp: main gui
	-$(CC) $(CFLAGS) $(OBJ) -o $(OUTPATH)/$(NAME) \
	-L$(QTLIBPATH) $(QTLIBS) -L$(INTELLIBPATH) $(INTELLIBS) \
	-Xlinker -rpath=$(CBMCORELIBPATHR) -Xlinker -rpath=$(CBMVISUALLIBPATHR) \
	-Xlinker --library-path=$(CBMCORELIBPATHR) -Xlinker --library-path=$(CBMVISUALLIBPATHR) \
	-Xlinker --library=$(CBMCORELIB) -Xlinker --library=$(CBMVISUALLIB)
	
main: $(MAININC) $(MAINSRC)
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(SRCPATH)/main.cpp -o $(OUTPATH)/main.obj
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(SRCPATH)/ecmanagement.cpp -o $(OUTPATH)/ecmanagement.obj
	
gui: $(GUIINC) $(GUISRC)
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(GUISP)/mainw.cpp -o $(OUTPATH)/mainw.obj
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(GUISP)/testpanel.cpp -o $(OUTPATH)/testpanel.obj
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(GUISP)/simthread.cpp -o $(OUTPATH)/simthread.obj

guiinc: uics mocs
	
mocs: $(MOCINC)
	-$(MOC) $(GUIIP)/mainw.h -o $(GUIMOCIP)/moc_mainw.h
	-$(MOC) $(GUIIP)/testpanel.h -o $(GUIMOCIP)/moc_testpanel.h
	-$(MOC) $(GUIIP)/simthread.h -o $(GUIMOCIP)/moc_simthread.h
	
uics: $(UIS)
	-$(UIC) $(GUIUIIP)/mainw.ui -o $(GUIUICIP)/ui_mainw.h
	-$(UIC) $(GUIUIIP)/testpanel.ui -o $(GUIUICIP)/ui_testpanel.h
	
clean:
	-$(RM) $(OUTPATH)/$(NAME)
	-$(RM) $(OBJ)
	
run: $(OUTPATH)/$(NAME)
	-$(OUTPATH)/$(NAME)
	