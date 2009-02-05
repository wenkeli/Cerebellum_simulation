#makes simulation
NAME = cbm

CC = g++
CFLAGS = #-W -Wall -pedantic-errors
RM = rm
MOC = moc
UIC = uic

QTINCPATH = '../../../Qt/4.4.3-mingw/include/'
QTLIBPATH = 'C:/Qt/4.4.3-mingw/lib/'
QTLIBS = -lqtmaind -lQTGuid4 -lQTCored4

INCLUDEPATH	 = ./includes
SRCPATH = ./src
OUTPUTPATH = ./output

UIS = $(INCLUDEPATH)/genesismw.ui $(INCLUDEPATH)/mfgrmainp.ui $(INCLUDEPATH)/mfgrconnsw.ui
MOCINC = $(INCLUDEPATH)/genesismw.h $(INCLUDEPATH)/mfgrmainp.h $(INCLUDEPATH)/mfgrconnsw.h

UICOUT = $(INCLUDEPATH)/ui_genesismw.h $(INCLUDEPATH)/ui_mfgrmainp.h $(INCLUDEPATH)/ui_mfgrconnsw.h
MOCOUT = $(INCLUDEPATH)/moc_genesismw.h $(INCLUDEPATH)/moc_mfgrmainp.h $(INCLUDEPATH)/moc_mfgrconnsw.h

INCLUDES = $(INCLUDEPATH)/common.h $(INCLUDEPATH)/parameters.h $(INCLUDEPATH)/randomc.h $(INCLUDEPATH)/synapsegenesis.h $(INCLUDEPATH)/main.h $(INCLUDEPATH)/globalvars.h $(INCLUDEPATH)/genesismw.h $(UICOUT) $(MOCOUT)

SRC = $(SRCPATH)/mother.cpp $(SRCPATH)/synapsegenesis.cpp $(SRCPATH)/main.cpp $(SRCPATH)/genesismw.cpp $(SRCPATH)/mfgrmainp.cpp $(SRCPATH)/mfgrconnsw.cpp
OBJ= $(OUTPUTPATH)/mother.o $(OUTPUTPATH)/synapsegenesis.o $(OUTPUTPATH)/main.o $(OUTPUTPATH)/genesismw.o $(OUTPUTPATH)/mfgrmainp.o $(OUTPUTPATH)/mfgrconnsw.o

all: objects
	-$(CC) $(CFLAGS) $(OBJ) -L $(QTLIBPATH) -o $(OUTPUTPATH)/$(NAME) $(QTLIBS)
objects: uics mocs $(INCLUDES) $(SRC)
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -c $(SRCPATH)/mother.cpp -o $(OUTPUTPATH)/mother.o
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -c $(SRCPATH)/synapsegenesis.cpp -o $(OUTPUTPATH)/synapsegenesis.o 
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -c $(SRCPATH)/main.cpp -o $(OUTPUTPATH)/main.o
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -c $(SRCPATH)/genesismw.cpp -o $(OUTPUTPATH)/genesismw.o
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -c $(SRCPATH)/mfgrmainp.cpp -o $(OUTPUTPATH)/mfgrmainp.o
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -c $(SRCPATH)/mfgrconnsw.cpp -o $(OUTPUTPATH)/mfgrconnsw.o
	
mocs: $(MOCINC)
	-$(MOC) $(INCLUDEPATH)/genesismw.h -o $(INCLUDEPATH)/moc_genesismw.h
	-$(MOC) $(INCLUDEPATH)/mfgrmainp.h -o $(INCLUDEPATH)/moc_mfgrmainp.h
	-$(MOC) $(INCLUDEPATH)/mfgrconnsw.h -o $(INCLUDEPATH)/moc_mfgrconnsw.h
uics: $(UIS)
	-$(UIC) $(INCLUDEPATH)/genesismw.ui -o $(INCLUDEPATH)/ui_genesismw.h
	-$(UIC) $(INCLUDEPATH)/mfgrmainp.ui -o $(INCLUDEPATH)/ui_mfgrmainp.h
	-$(UIC) $(INCLUDEPATH)/mfgrconnsw.ui -o $(INCLUDEPATH)/ui_mfgrconnsw.h

clean: fclean
	$(RM) $(OUTPUTPATH)/$(NAME).exe
fclean:
	$(RM) $(OBJ) $(UICOUT) $(MOCOUT)

run: $(OUTPUTPATH)/$(NAME).exe
	-$(OUTPUTPATH)/$(NAME)
	