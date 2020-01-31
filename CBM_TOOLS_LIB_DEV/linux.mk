NAME = libcbm_tools

CC = icpc

DEFINES = -DINTELCC

CFLAGS = $(DEFINES) -O3 -fpic 
#-openmp

RM = rm

CXXTOOLSIP = '../CXX_TOOLS_LIB/'
CBMDATAIP = '../CBM_DATA_LIB'
EXTINCPATH = -I $(CXXTOOLSIP) -I $(CBMDATAIP)  

DEPLIBPATH = ../libs
DEPLIBS = -Xlinker --library=cxx_tools -Xlinker --library=cbm_data

INCPATH = ./CBMToolsInclude

SRCPATH = ./src

OUTPATH = ./intout
LIBPATH = ./lib

INC = $(INCPATH)/poissonregencells.h $(INCPATH)/eyelidintegrator.h $(INCPATH)/ecmfpopulation.h

SRC = $(SRCPATH)/poissonregencells.cpp $(SRCPATH)/eyelidintegrator.cpp $(SRCPATH)/ecmfpopulation.cpp

OBJ = $(OUTPATH)/poissonregencells.obj $(OUTPATH)/eyelidintegrator.obj $(OUTPATH)/ecmfpopulation.obj

lib: obj
	-$(CC) $(CFLAGS) -G $(OBJ) -o $(LIBPATH)/$(NAME).so \
	-Xlinker -rpath=$(DEPLIBPATH) -Xlinker --library-path=$(DEPLIBPATH) \
	$(DEPLIBS)
	-ln -sfn ../CBM_TOOLS_LIB/$(LIBPATH)/$(NAME).so $(DEPLIBPATH)/
	
obj: $(INC) $(SRC) 
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(SRCPATH)/poissonregencells.cpp -o $(OUTPATH)/poissonregencells.obj
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(SRCPATH)/eyelidintegrator.cpp -o $(OUTPATH)/eyelidintegrator.obj
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(SRCPATH)/ecmfpopulation.cpp -o $(OUTPATH)/ecmfpopulation.obj
	
cleanall: cleanlib cleanobj
	
cleanobj:
	-$(RM) $(OBJ)
	
cleanlib:
	-$(RM) $(LIBPATH)/$(NAME).so

