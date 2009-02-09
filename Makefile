#makes simulation
NAME = cbm

CC = g++
CFLAGS = #-W -Wall -pedantic-errors
RM = rm
MOC = moc
UIC = uic

QTINCPATH = 'C:/Qt/4.4.3-mingw/include/'
QTLIBPATH = 'C:/Qt/4.4.3-mingw/lib/'
QTLIBS = -lqtmaind -lQTGuid4 -lQTCored4

INCLUDEPATH	 = ./includes
SRCPATH = ./src
OUTPUTPATH = ./output

UIS = $(INCLUDEPATH)/genesismw.ui $(INCLUDEPATH)/dispdialoguep.ui $(INCLUDEPATH)/conndispw.ui
MOCINC = $(INCLUDEPATH)/genesismw.h $(INCLUDEPATH)/dispdialoguep.h $(INCLUDEPATH)/conndispw.h

UICOUT = $(INCLUDEPATH)/ui_genesismw.h $(INCLUDEPATH)/ui_dispdialoguep.h $(INCLUDEPATH)/ui_conndispw.h
MOCOUT = $(INCLUDEPATH)/moc_genesismw.h $(INCLUDEPATH)/moc_dispdialoguep.h $(INCLUDEPATH)/moc_conndispw.h

INCLUDES = $(INCLUDEPATH)/common.h $(INCLUDEPATH)/parameters.h $(INCLUDEPATH)/randomc.h $(INCLUDEPATH)/synapsegenesis.h $(INCLUDEPATH)/main.h $(INCLUDEPATH)/globalvars.h $(MOCINC) $(UICOUT) $(MOCOUT)

SRC = $(SRCPATH)/mother.cpp $(SRCPATH)/synapsegenesis.cpp $(SRCPATH)/main.cpp $(SRCPATH)/genesismw.cpp $(SRCPATH)/dispdialoguep.cpp $(SRCPATH)/conndispw.cpp
OBJ= $(OUTPUTPATH)/mother.o $(OUTPUTPATH)/synapsegenesis.o $(OUTPUTPATH)/main.o $(OUTPUTPATH)/genesismw.o $(OUTPUTPATH)/dispdialoguep.o $(OUTPUTPATH)/conndispw.o

all: objects
	-$(CC) $(CFLAGS) $(OBJ) -L $(QTLIBPATH) -o $(OUTPUTPATH)/$(NAME) $(QTLIBS)
objects: uics mocs $(INCLUDES) $(SRC)
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -c $(SRCPATH)/mother.cpp -o $(OUTPUTPATH)/mother.o
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -c $(SRCPATH)/synapsegenesis.cpp -o $(OUTPUTPATH)/synapsegenesis.o 
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -c $(SRCPATH)/main.cpp -o $(OUTPUTPATH)/main.o
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -c $(SRCPATH)/genesismw.cpp -o $(OUTPUTPATH)/genesismw.o
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -c $(SRCPATH)/dispdialoguep.cpp -o $(OUTPUTPATH)/dispdialoguep.o
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -c $(SRCPATH)/conndispw.cpp -o $(OUTPUTPATH)/conndispw.o
	
mocs: $(MOCINC)
	-$(MOC) $(INCLUDEPATH)/genesismw.h -o $(INCLUDEPATH)/moc_genesismw.h
	-$(MOC) $(INCLUDEPATH)/dispdialoguep.h -o $(INCLUDEPATH)/moc_dispdialoguep.h
	-$(MOC) $(INCLUDEPATH)/conndispw.h -o $(INCLUDEPATH)/moc_conndispw.h
uics: $(UIS)
	-$(UIC) $(INCLUDEPATH)/genesismw.ui -o $(INCLUDEPATH)/ui_genesismw.h
	-$(UIC) $(INCLUDEPATH)/dispdialoguep.ui -o $(INCLUDEPATH)/ui_dispdialoguep.h
	-$(UIC) $(INCLUDEPATH)/conndispw.ui -o $(INCLUDEPATH)/ui_conndispw.h

clean: fclean
	$(RM) $(OUTPUTPATH)/$(NAME).exe
fclean:
	$(RM) $(OBJ) $(UICOUT) $(MOCOUT)

run: $(OUTPUTPATH)/$(NAME).exe
	-$(OUTPUTPATH)/$(NAME)
	