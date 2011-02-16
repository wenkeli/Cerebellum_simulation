
NAME = BayesianSVMF

CC = icpc
DEFINES = -DINTELCC
CFLAGS = $(DEFINES) -O3

RM = rm

INCPATH = ./includes
SRCPATH = ./src
OUTPATH = ./output

COMMONINCS = $(INCPATH)/globalvars.h $(INCPATH)/main.h $(INCPATH)/parameters.h
MATHINCS = $(INCPATH)/bayesianeval.h $(INCPATH)/mfactivities.h $(INCPATH)/randomc.h $(INCPATH)/sfmt.h
IOINCS = $(INCPATH)/readinputs.h $(INCPATH)/writeoutputs.h
INCS = $(COMMONINCS) $(MATHINCS) $(IOINCS)

SRC = $(SRCPATH)/bayesianeval.cpp $(SRCPATH)/main.cpp $(SRCPATH)/mfactivities.cpp $(SRCPATH)/readinputs.cpp $(SRCPATH)/sfmt.cpp $(SRCPATH)/writeoutputs.cpp

OBJ = $(OUTPATH)/bayesianeval.obj $(OUTPATH)/main.obj $(OUTPATH)/mfactivities.obj $(OUTPATH)/readinputs.obj $(OUTPATH)/sfmt.obj $(OUTPATH)/writeoutputs.obj

all: objs $(OBJ)
	-$(CC) $(CFLAGS) $(OBJ) -o $(OUTPATH)/$(NAME)
	
objs: $(INCS) $(SRC)
	-$(CC) $(CFLAGS) -c $(SRCPATH)/bayesianeval.cpp -o$(OUTPATH)/bayesianeval.obj
	-$(CC) $(CFLAGS) -c $(SRCPATH)/main.cpp -o$(OUTPATH)/main.obj
	-$(CC) $(CFLAGS) -c $(SRCPATH)/mfactivities.cpp -o$(OUTPATH)/mfactivities.obj
	-$(CC) $(CFLAGS) -c $(SRCPATH)/readinputs.cpp -o$(OUTPATH)/readinputs.obj
	-$(CC) $(CFLAGS) -c $(SRCPATH)/sfmt.cpp -o$(OUTPATH)/sfmt.obj
	-$(CC) $(CFLAGS) -c $(SRCPATH)/writeoutputs.cpp -o$(OUTPATH)/writeoutputs.obj

clean: fclean
	-$(RM) $(OUTPATH)/$(NAME)
	
fclean:
	-$(RM) $(OBJ)
	