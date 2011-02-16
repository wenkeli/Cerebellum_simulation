
NAME = MFInputGen

cc = icc
DEFINES =
CFLAGS = $(DEFINES) -O3

RM = rm

INCPATH = ./includes
SRCPATH = ./src
OUTPATH = ./output

INCS = $(INCPATH)/main.h $(INCPATH)/readin.h $(INCPATH)/globalvars.h

SRC = $(SRCPATH)/main.cpp $(SRCPATH)/readin.cpp

OBJ = $(OUTPATH)/main.obj $(OUTPATH)/readin.obj

all: objs $(OBJ)
	-$(CC) $(CFLAGS) $(OBJ) -o $(OUTPATH)/$(NAME)

objs: $(INCS) $(SRC)
	-$(CC) $(CFLAGS) -c $(SRCPATH)/main.cpp -o$(OUTPATH)/main.obj
	-$(CC) $(CFLAGS) -c $(SRCPATH)/readin.cpp -o$(OUTPATH)/readin.obj

clean: fclean
	-$(RM) $(OUTPATH)/$(NAME)

fclean:
	-$(RM) $(OBJ)	