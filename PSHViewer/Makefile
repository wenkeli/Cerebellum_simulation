#makes IO single cell simulation for subthreshold oscillation parameters exploration

#-DDEBUG -DINTELCC

NAME = PSHViewer


CC = icl
DEFINES = 
CFLAGS = $(DEFINES) -nologo -fast -O2 -Oa -Qvec-guard-write -Qopenmp -Qipo -arch:SSE4.1 -F100000000 -MD

RM = rm
MOC = moc
UIC = uic

INCPATH = ./includes
SRCPATH = ./src
OUTPATH = ./output

QTINCPATH = 'C:/Qt/4.6.3/include/'
QTLIBPATH = 'C:/Qt/4.6.3/lib/'
QTLIBS = -DEFAULTLIB:qtmain -DEFAULTLIB:QTGui4 -DEFAULTLIB:QTCore4

UIS = $(INCPATH)/mainw.ui $(INCPATH)/pshdispw.ui
MOCINC = $(INCPATH)/mainw.h $(INCPATH)/pshdispw.h
UICOUT = $(INCPATH)/ui_mainw.h $(INCPATH)/ui_pshdispw.h
MOCOUT = $(INCPATH)/moc_mainw.h $(INCPATH)/moc_pshdispw.h
COMINCS = $(INCPATH)/common.h
COREINCS = $(INCPATH)/main.h
INCS = $(MOCINC) $(UICOUT) $(MOCOUT) $(COMINCS) $(GUIINCS) $(COREINCS)

GUISRC = $(SRCPATH)/mainw.cpp $(SRCPATH)/pshdispw.cpp
CORESRC = $(SRCPATH)/main.cpp
SRC = $(GUISRC) $(CORESRC)

GUIOBJ = $(OUTPATH)/mainw.obj $(OUTPATH)/pshdispw.obj
COREOBJ = $(OUTPATH)/main.obj
OBJ = $(GUIOBJ) $(COREOBJ)

all: core gui
	-$(CC) $(CFLAGS) $(OBJ) -Fe$(OUTPATH)/$(NAME) -link -RELEASE -LIBPATH:$(QTLIBPATH) $(QTLIBS)

core: mocs uis $(INCS) $(CORESRC)
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -c $(SRCPATH)/main.cpp -Fo$(OUTPATH)/main.obj
	
gui: mocs uis $(INC) $(GUISRC)
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -c $(SRCPATH)/mainw.cpp -Fo$(OUTPATH)/mainw.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -c $(SRCPATH)/pshdispw.cpp -Fo$(OUTPATH)/pshdispw.obj

mocs:
	-$(MOC) $(INCPATH)/mainw.h -o $(INCPATH)/moc_mainw.h
	-$(MOC) $(INCPATH)/pshdispw.h -o $(INCPATH)/moc_pshdispw.h
	
uis:
	-$(UIC) $(INCPATH)/mainw.ui -o $(INCPATH)/ui_mainw.h
	-$(UIC) $(INCPATH)/pshdispw.ui -o $(INCPATH)/ui_pshdispw.h
	
clean: fclean
	-$(RM) $(OUTPATH)/$(NAME).exe $(OUTPATH)/$(NAME).exe.manifest

fclean:
	-$(RM) $(OBJ) $(UICOUT) $(MOCOUT)

run: $(OUTPATH)/$(NAME).exe
	-$(OUTPATH)/$(NAME)
	