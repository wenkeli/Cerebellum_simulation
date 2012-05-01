#makes IO single cell simulation for subthreshold oscillation parameters exploration

#-DDEBUG -DINTELCC
#

NAME = SimDataAnalysis


CC = g++
#icpc
DEFINES = -DEYELID
#-DINTELCC
#-DCARTPOLE

CFLAGS = $(DEFINES) -O3 -openmp -mcmodel=medium
#-mcmodel medium -shared-intel -std=c++0x

RM = rm
MOC = moc
UIC = uic

QTINCPATH = '/usr/include/qt4/'
QTLIBPATH = '/usr/lib/qt4/'
QTLIBS = -lQtGui -lQtCore

GSLINCPATH = '/usr/include/gsl/'
GSLLIBPATH = '/usr/lib/'
GSLLIBS = -lgsl -lgslcblas

EXTINCPATH = -I $(QTINCPATH) -I $(GSLINCPATH) 

INCPATH = ./includes
DAMIP = $(INCPATH)/datamodules
ANMIP = $(INCPATH)/analysismodules

SRCPATH = ./src
DAMSP = $(SRCPATH)/datamodules
ANMSP = $(SRCPATH)/analysismodules

OUTPATH = ./output

UIS = $(INCPATH)/mainw.ui $(INCPATH)/pshdispw.ui
MOCINC = $(INCPATH)/mainw.h $(INCPATH)/pshdispw.h
UICOUT = $(INCPATH)/ui_mainw.h $(INCPATH)/ui_pshdispw.h
MOCOUT = $(INCPATH)/moc_mainw.h $(INCPATH)/moc_pshdispw.h
COMINCS = $(INCPATH)/common.h $(INCPATH)/globalvars.h
COREINCS = $(INCPATH)/main.h $(INCPATH)/randomc.h $(INCPATH)/sfmt.h

DAMINCS = $(DAMIP)/psh.h $(DAMIP)/pshgpu.h $(DAMIP)/simerrorec.h $(DAMIP)/simexternalec.h $(DAMIP)/siminnet.h $(DAMIP)/simmfinputec.h $(DAMIP)/simmzone.h $(DAMIP)/simoutputec.h
ANMINCS = $(ANMIP)/grpshpopanalysis.h $(ANMIP)/grconpshanalysis.h $(ANMIP)/spikerateanalysis.h \
$(ANMIP)/pshtravclusterbase.h $(ANMIP)/pshtravclusterpos2st.h $(ANMIP)/pshtravclustereucdist.h \
$(ANMIP)/innetspatialvis.h

INCS = $(MOCINC) $(UICOUT) $(MOCOUT) $(COMINCS) $(GUIINCS) $(COREINCS) $(DAMINCS) $(ANMINCS)

GUISRC = $(SRCPATH)/mainw.cpp $(SRCPATH)/pshdispw.cpp
CORESRC = $(SRCPATH)/main.cpp $(SRCPATH)/sfmt.cpp

DAMSRC = $(DAMSP)/psh.cpp $(DAMSP)/pshgpu.cpp $(DAMSP)/simerrorec.cpp $(DAMSP)/simexternalec.cpp $(DAMSP)/siminnet.cpp $(DAMSP)/simmfinputec.cpp $(DAMSP)/simmzone.cpp $(DAMSP)/simoutputec.cpp
ANMSRC = $(ANMSP)/grpshpopanalysis.cpp $(ANMSP)/grconpshanalysis.cpp $(ANMSP)/spikerateanalysis.cpp \
$(ANMSP)/pshtravclusterbase.cpp $(ANMSP)/pshtravclusterpos2st.cpp $(ANMSP)/pshtravclustereucdist.cpp \
$(ANMSP)/innetspatialvis.cpp

SRC = $(GUISRC) $(CORESRC) $(DAMSRC) $(ANMSRC)

GUIOBJ = $(OUTPATH)/mainw.obj $(OUTPATH)/pshdispw.obj
COREOBJ = $(OUTPATH)/main.obj $(OUTPATH)/sfmt.obj

DAMOBJ = $(OUTPATH)/psh.obj $(OUTPATH)/pshgpu.obj $(OUTPATH)/simerrorec.obj $(OUTPATH)/simexternalec.obj $(OUTPATH)/siminnet.obj $(OUTPATH)/simmfinputec.obj $(OUTPATH)/simmzone.obj $(OUTPATH)/simoutputec.obj
ANMOBJ = $(OUTPATH)/grpshpopanalysis.obj $(OUTPATH)/grconpshanalysis.obj $(OUTPATH)/spikerateanalysis.obj \
$(OUTPATH)/pshtravclusterbase.obj $(OUTPATH)/pshtravclusterpos2st.obj $(OUTPATH)/pshtravclustereucdist.obj \
$(OUTPATH)/innetspatialvis.obj

OBJ = $(GUIOBJ) $(COREOBJ) $(DAMOBJ) $(ANMOBJ)

all: core gui dam anm
	-$(CC) $(CFLAGS) $(OBJ) -o $(OUTPATH)/$(NAME) -L$(QTLIBPATH) $(QTLIBS) -L$(GSLLIBPATH) $(GSLLIBS)

core: $(INCS) $(CORESRC)
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(SRCPATH)/main.cpp -o$(OUTPATH)/main.obj
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(SRCPATH)/sfmt.cpp -o$(OUTPATH)/sfmt.obj
	
gui: $(INC) $(GUISRC)
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(SRCPATH)/mainw.cpp -o$(OUTPATH)/mainw.obj
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(SRCPATH)/pshdispw.cpp -o$(OUTPATH)/pshdispw.obj
	
dam: $(INC) $(DAMSRC)
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(DAMSP)/psh.cpp -o$(OUTPATH)/psh.obj
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(DAMSP)/pshgpu.cpp -o$(OUTPATH)/pshgpu.obj
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(DAMSP)/simerrorec.cpp -o$(OUTPATH)/simerrorec.obj
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(DAMSP)/simexternalec.cpp -o$(OUTPATH)/simexternalec.obj
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(DAMSP)/siminnet.cpp -o$(OUTPATH)/siminnet.obj
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(DAMSP)/simmfinputec.cpp -o$(OUTPATH)/simmfinputec.obj
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(DAMSP)/simmzone.cpp -o$(OUTPATH)/simmzone.obj
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(DAMSP)/simoutputec.cpp -o$(OUTPATH)/simoutputec.obj
	
anm: $(INC) $(ANMSRC)
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(ANMSP)/grpshpopanalysis.cpp -o$(OUTPATH)/grpshpopanalysis.obj
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(ANMSP)/grconpshanalysis.cpp -o$(OUTPATH)/grconpshanalysis.obj
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(ANMSP)/spikerateanalysis.cpp -o$(OUTPATH)/spikerateanalysis.obj
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(ANMSP)/pshtravclusterbase.cpp -o$(OUTPATH)/pshtravclusterbase.obj
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(ANMSP)/pshtravclusterpos2st.cpp -o$(OUTPATH)/pshtravclusterpos2st.obj
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(ANMSP)/pshtravclustereucdist.cpp -o$(OUTPATH)/pshtravclustereucdist.obj
	-$(CC) $(CFLAGS) $(EXTINCPATH) -c $(ANMSP)/innetspatialvis.cpp -o$(OUTPATH)/innetspatialvis.obj
	
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
	