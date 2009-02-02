#makes simulation
NAME = cbm

CC = g++
CFLAGS = -W -Wall -pedantic-errors
RM = rm

INCLUDEPATH	 = ./includes
SRCPATH = ./src
OUTPUTPATH = ./output

INCLUDES = $(INCLUDEPATH)/common.h $(INCLUDEPATH)/parameters.h $(INCLUDEPATH)/randomc.h $(INCLUDEPATH)/synapsegenesis.h $(INCLUDEPATH)/main.h $(INCLUDEPATH)/globalvars.h
SRC = $(SRCPATH)/mother.cpp $(SRCPATH)/synapsegenesis.cpp $(SRCPATH)/main.cpp
OBJ= $(OUTPUTPATH)/mother.o $(OUTPUTPATH)/synapsegenesis.o $(OUTPUTPATH)/main.o

all: objects
	-$(CC) $(CFLAGS) $(OBJ) -o $(OUTPUTPATH)/$(NAME)
objects: $(INCLUDES) $(SRC)
	-$(CC) $(CFLAGS) -c $(SRCPATH)/mother.cpp -o $(OUTPUTPATH)/mother.o
	-$(CC) $(CFLAGS) -c $(SRCPATH)/synapsegenesis.cpp -o $(OUTPUTPATH)/synapsegenesis.o 
	-$(CC) $(CFLAGS) -c $(SRCPATH)/main.cpp -o $(OUTPUTPATH)/main.o

clean:
	$(RM) $(OBJ) $(OUTPUTPATH)/$(NAME).exe
fclean:
	$(RM) $(OBJ)