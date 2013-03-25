NAME = Sim_Analysis

CC = clang++
#icpc
DEFINES = 
#-DINTELCC

CFLAGS = $(DEFINES) -O3 
#-mcmodel medium -shared-intel

RM = rm
MOC = /usr/local/Trolltech/Qt-4.7.2/bin/moc
UIC = /usr/local/Trolltech/Qt-4.7.2/bin/uic

QTIP = '/usr/local/Trolltech/Qt-4.7.2/include'
CXXTOOLSIP = '../CXX_TOOLS_LIB'
CBMSTATEIP = '../CBM_STATE_LIB'
CBMDATAIP = '../CBM_DATA_LIB'
CBMVISUALIP = '../CBM_VISUAL_LIB'
EXTINCPATH = -I $(QTIP) -I $(CXXTOOLSIP) -I $(CBMSTATEIP) -I $(CBMDATAIP) -I $(CBMVISUALIP)

QTLIBPATH = '/usr/local/Trolltech/Qt-4.7.2/lib/'
QTLIBS = -lQtGui -lQtCore

DEPLIBPATH = ../libs/
DEPLIBS = -Xlinker --library=cxx_tools -Xlinker --library=cbm_state -Xlinker --library=cbm_data \
-Xlinker --library=cbm_visual 

INCPATH = ./includes
GUIIP = $(INCPATH)/gui
GUIMOCIP = $(GUIIP)/moc
GUIUICIP = $(GUIIP)/uic
GUIUIIP = $(GUIIP)/ui

SRCPATH = ./src
GUISP = $(SRCPATH)/gui

OUTPATH = ./output

UIS =  $(GUIUIIP)/mainw.ui
MOCINC = $(GUIIP)/mainw.h
UICOUT = $(GUIUICIP)/ui_mainw.h
MOCOUT = $(GUIMOCIP)/moc_mainw.h
GUIINC = $(MOCINC) $(MOCOUT) $(UICOUT)

MAININC = $(INCPATH)/main.h 

INC = $(GUIINC) $(MAININC)

GUISRC = $(GUISP)/mainw.cpp
MAINSRC = $(SRCPATH)/main.cpp

SRC= $(GUISRC) $(MAINSRC)

GUIOBJ = $(OUTPATH)/mainw.obj
MAINOBJ = $(OUTPATH)/main.obj
OBJ = $(GUIOBJ) $(MAINOBJ)

mainapp: main gui
	-$(CC) $(CFLAGS) $(OBJ) -o $(OUTPATH)/$(NAME) \
	-L$(QTLIBPATH) $(QTLIBS) \
	-Xlinker -rpath=$(DEPLIBPATH) -Xlinker --library-path=$(DEPLIBPATH) \
	$(DEPLIBS)
	
main: $(MAININC) $(MAINSRC)
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(SRCPATH)/main.cpp -o $(OUTPATH)/main.obj
	
gui: guiinc $(GUIINC) $(GUISRC)
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(GUISP)/mainw.cpp -o $(OUTPATH)/mainw.obj

guiinc: uics mocs
	
mocs: $(MOCINC)
	-$(MOC) $(GUIIP)/mainw.h -o $(GUIMOCIP)/moc_mainw.h
	
uics: $(UIS)
	-$(UIC) $(GUIUIIP)/mainw.ui -o $(GUIUICIP)/ui_mainw.h
	
clean:
	-$(RM) $(OBJ) $(OUTPATH)/$(NAME)
	
