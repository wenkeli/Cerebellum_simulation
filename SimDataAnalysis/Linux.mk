#makes IO single cell simulation for subthreshold oscillation parameters exploration

#-DDEBUG -DINTELCC
#

NAME = SimDataAnalysis


CC = icpc
DEFINES = -DEYELID
#-DCARTPOLE

CFLAGS = $(DEFINES) -O3 -openmp -mcmodel medium -shared-intel

RM = rm
MOC = moc
UIC = uic

QTINCPATH = '/usr/include/qt4/'
QTLIBPATH = '/usr/lib/qt4/'
QTLIBS = -lQtGui -lQtCore

INCPATH = ./includes
DAMIP = $(INCPATH)/datamodules

SRCPATH = ./src
DAMSP = $(SRCPATH)/datamodules

OUTPATH = ./output

UIS = $(INCPATH)/mainw.ui $(INCPATH)/pshdispw.ui
MOCINC = $(INCPATH)/mainw.h $(INCPATH)/pshdispw.h
UICOUT = $(INCPATH)/ui_mainw.h $(INCPATH)/ui_pshdispw.h
MOCOUT = $(INCPATH)/moc_mainw.h $(INCPATH)/moc_pshdispw.h
COMINCS = $(INCPATH)/common.h
COREINCS = $(INCPATH)/main.h

DAMINCS = $(DAMIP)/psh.h $(DAMIP)/pshgpu.h

INCS = $(MOCINC) $(UICOUT) $(MOCOUT) $(COMINCS) $(GUIINCS) $(COREINCS) $(DAMINCS)

GUISRC = $(SRCPATH)/mainw.cpp $(SRCPATH)/pshdispw.cpp
CORESRC = $(SRCPATH)/main.cpp

DAMSRC = $(DAMSP)/psh.cpp $(DAMSP)/pshgpu.cpp

SRC = $(GUISRC) $(CORESRC) $(DAMSRC)

GUIOBJ = $(OUTPATH)/mainw.obj $(OUTPATH)/pshdispw.obj
COREOBJ = $(OUTPATH)/main.obj

DAMOBJ = $(OUTPATH)/psh.obj $(OUTPATH)/pshgpu.obj

OBJ = $(GUIOBJ) $(COREOBJ) $(DAMOBJ)

all: core gui dam
	-$(CC) $(CFLAGS) $(OBJ) -o $(OUTPATH)/$(NAME) -L$(QTLIBPATH) $(QTLIBS)

core: $(INCS) $(CORESRC)
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -c $(SRCPATH)/main.cpp -o$(OUTPATH)/main.obj
	
gui: $(INC) $(GUISRC)
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -c $(SRCPATH)/mainw.cpp -o$(OUTPATH)/mainw.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -c $(SRCPATH)/pshdispw.cpp -o$(OUTPATH)/pshdispw.obj
	
dam: $(INC) $(DAMSRC)
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -c $(DAMSP)/psh.cpp -o$(OUTPATH)/psh.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -c $(DAMSP)/pshgpu.cpp -o$(OUTPATH)/pshgpu.obj

guiinc: mocs uis
	
mocs:
	-$(MOC) $(INCPATH)/mainw.h -o $(INCPATH)/moc_mainw.h
	-$(MOC) $(INCPATH)/pshdispw.h -o $(INCPATH)/moc_pshdispw.h
	
uis:
	-$(UIC) $(INCPATH)/mainw.ui -o $(INCPATH)/ui_mainw.h
	-$(UIC) $(INCPATH)/pshdispw.ui -o $(INCPATH)/ui_pshdispw.h
	
clean: fclean
	-$(RM) $(OUTPATH)/$(NAME)

fclean:
	-$(RM) $(OBJ)

run: $(OUTPATH)/$(NAME)
	-$(OUTPATH)/$(NAME)
	