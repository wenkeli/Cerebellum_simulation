#eyelid conditioning simulation

NAME = cbm_eyelid

CC = icpc
NVCC = nvcc

DEFINES = -DINTELCC

CFLAGS = $(DEFINES) -O3

RM = rm
MOC = moc
UIC = uic

CUDAIP = '/opt/cuda/include/'
QTIP = '/usr/include/qt4/'

CXXTOOLSIP = '../CXX_TOOLS_LIB'
CBMTOOLSIP = '../CBM_TOOLS_LIB'
CBMSTATEIP = '../CBM_STATE_LIB'
CBMCOREIP = '../CBM_CORE_LIB'
CBMVISUALIP = '../CBM_VISUAL_LIB'
CBMDATAIP = '../CBM_DATA_LIB'
EXTINCPATH = -I $(CUDAIP) -I $(QTIP) -I $(CXXTOOLSIP) -I $(CBMTOOLSIP) -I $(CBMSTATEIP) -I $(CBMCOREIP) \
-I $(CBMVISUALIP) -I $(CBMDATAIP)

QTLIBPATH = '/usr/lib/qt4'
QTLIBS = -lQtGui -lQtCore

DEPLIBPATH = ../libs/
DEPLIBS = -Xlinker --library=cxx_tools -Xlinker --library=cbm_tools -Xlinker --library=cbm_state \
-Xlinker --library=cbm_core -Xlinker --library=cbm_visual -Xlinker --library=cbm_data


INCPATH = ./includes
GUIIP = $(INCPATH)/gui
GUIMOCIP = $(GUIIP)/moc
GUIUICIP = $(GUIIP)/uic
GUIUIIP = $(GUIIP)/ui

ECTRIALIP = $(INCPATH)/ectrial

SRCPATH = ./src
GUISP = $(SRCPATH)/gui
ECTRIALSP = $(SRCPATH)/ectrial

OUTPATH = ./output

UIS = $(GUIUIIP)/mainw.ui $(GUIUIIP)/testpanel.ui
MOCINC = $(GUIIP)/mainw.h $(GUIIP)/testpanel.h $(GUIIP)/simthread.h
UICOUT = $(GUIUICIP)/ui_mainw.h $(GUIUICIP)/ui_testpanel.h
MOCOUT = $(GUIMOCIP)/moc_mainw.h $(GUIMOCIP)/moc_testpanel.h $(GUIMOCIP)/moc_simthread.h
GUIINC = $(MOCINC) $(MOCOUT) $(UICOUT)

ECTRIALINC = $(ECTRIALIP)/ecmanagementbase.h $(ECTRIALIP)/ecmanagementdelay.h

MAININC= $(INCPATH)/main.h $(INCPATH)/interthreadcomm.h

INC = $(GUIINC) $(ECTRIALINC) $(MAININC)

GUISRC = $(GUISP)/mainw.cpp $(GUISP)/testpanel.cpp $(GUISP)/simthread.cpp
ECTRIALSRC = $(ECTRIALSP)/ecmanagementbase.cpp $(ECTRIALSP)/ecmanagementdelay.cpp
MAINSRC = $(SRCPATH)/main.cpp $(SRCPATH)/interthreadcomm.cpp

SRC = $(GUISRC) $(ECTRIALSRC) $(MAINSRC)

GUIOBJ = $(OUTPATH)/mainw.obj $(OUTPATH)/testpanel.obj $(OUTPATH)/simthread.obj
ECTRIALOBJ = $(OUTPATH)/ecmanagementbase.obj $(OUTPATH)/ecmanagementdelay.obj
MAINOBJ = $(OUTPATH)/main.obj $(OUTPATH)/interthreadcomm.obj

OBJ = $(GUIOBJ) $(ECTRIALOBJ) $(MAINOBJ)

mainapp: main gui ectrial
	-$(CC) $(CFLAGS) $(OBJ) -o $(OUTPATH)/$(NAME) \
	-L$(QTLIBPATH) $(QTLIBS) \
	-Xlinker -rpath=$(DEPLIBPATH) -Xlinker --library-path=$(DEPLIBPATH) \
	$(DEPLIBS)
	
main: $(MAININC) $(MAINSRC)
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(SRCPATH)/main.cpp -o $(OUTPATH)/main.obj
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(SRCPATH)/interthreadcomm.cpp -o $(OUTPATH)/interthreadcomm.obj
	
ectrial: $(ECTRIALINC) $(ECTRIALSRC)
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(ECTRIALSP)/ecmanagementbase.cpp -o $(OUTPATH)/ecmanagementbase.obj
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(ECTRIALSP)/ecmanagementdelay.cpp -o $(OUTPATH)/ecmanagementdelay.obj
	
gui: guiinc $(GUIINC) $(GUISRC)
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
	