
NAME = MFInputGen

CC = icpc
DEFINES =
CFLAGS = $(DEFINES) -O3

RM = rm

INCPATH = ./includes
SRCPATH = ./src
OUTPATH = ./output

INCS = $(INCPATH)/main.h $(INCPATH)/readin.h $(INCPATH)/globalvars.h $(INCPATH)/randomc.h $(INCPATH)/sfmt.h $(INCPATH)/writeout.h

SRC = $(SRCPATH)/main.cpp $(SRCPATH)/readin.cpp $(SRCPATH)/sfmt.cpp $(SRCPATH)/writeout.cpp

OBJ = $(OUTPATH)/readin.obj $(OUTPATH)/main.obj $(OUTPATH)/sfmt.obj $(OUTPATH)/writeout.obj

all: objs $(OBJ)
	-$(CC) $(CFLAGS) $(OBJ) -o $(OUTPATH)/$(NAME)

objs: $(INCS) $(SRC)
	-$(CC) $(CFLAGS) -c $(SRCPATH)/main.cpp -o$(OUTPATH)/main.obj
	-$(CC) $(CFLAGS) -c $(SRCPATH)/readin.cpp -o$(OUTPATH)/readin.obj
	-$(CC) $(CFLAGS) -c $(SRCPATH)/sfmt.cpp -o$(OUTPATH)/sfmt.obj
	-$(CC) $(CFLAGS) -c $(SRCPATH)/writeout.cpp -o$(OUTPATH)/writeout.obj

clean: fclean
	-$(RM) $(OUTPATH)/$(NAME)

fclean:
	-$(RM) $(OBJ)
	