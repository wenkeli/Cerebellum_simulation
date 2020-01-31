#makes simulation -FIXED:no -Qparallel -Qopenmp_profile -Zi -debug:all -fixed:no -DDEBUG -DGPUDEBUG  -DACTDEBUG
NAME = cbm_new_CUDA

CC = g++
NVCC = nvcc
DEFINES = -DCARTPOLE -DVIZCP
#-DDEBUG
CFLAGS = $(DEFINES) -O3
NVCFLAGS = -arch=compute_20 
#-code=sm_20
# -MD  -Qvc8 -Qms2 -Wall -Qvec-guard-write -Qopt-streaming-stores:always -Qopenmp  -Qipo -fast -MD  -O3

RM = rm -f
MOC = moc
UIC = uic

INTELLIBPATH = '/opt/intel/lib/intel64'
INTELLIBS =

CUDAINCPATH = '/usr/local/cuda/include/'
CUDALIBPATH = '/usr/local/cuda/lib64/'
CUDALIBS = -lcudart 
#-lcudart

QTINCPATH = '/usr/local/Trolltech/Qt-4.7.2/include'
QTLIBPATH = '/usr/local/Trolltech/Qt-4.7.2/lib/'
QTLIBS = -lQtGui -lQtCore

SDLLIBS = -lSDL -lSDL_gfx -lSDL_ttf -l SDL_image

INCPATH = ./includes
GUIIP = $(INCPATH)/gui
CUDAIP = $(INCPATH)/cuda
MFMIP = $(INCPATH)/mfinputmodules
ERRMIP = $(INCPATH)/errorinputmodules
OUTMIP = $(INCPATH)/outputmodules
EXTMIP = $(INCPATH)/externalmodules
MZMIP = $(INCPATH)/mzonemodules
INMIP = $(INCPATH)/innetmodules
ANMIP = $(INCPATH)/analysismodules
 
SRCPATH = ./src
GUISP = $(SRCPATH)/gui
CUDASP = $(SRCPATH)/cuda
MFMSP = $(SRCPATH)/mfinputmodules
ERRMSP = $(SRCPATH)/errorinputmodules
OUTMSP = $(SRCPATH)/outputmodules
EXTMSP = $(SRCPATH)/externalmodules
MZMSP = $(SRCPATH)/mzonemodules
INMSP = $(SRCPATH)/innetmodules
ANMSP = $(SRCPATH)/analysismodules

EXTERN = ./cp_rl_comparison

OUTPATH = ./output

UIS = $(GUIIP)/mainw.ui $(GUIIP)/dispdialoguep.ui $(GUIIP)/conndispw.ui $(GUIIP)/simdispw.ui $(GUIIP)/simctrlp.ui $(GUIIP)/actdiagw.ui $(GUIIP)/spikeratesdispw.ui
MOCINC = $(GUIIP)/mainw.h $(GUIIP)/dispdialoguep.h $(GUIIP)/conndispw.h $(GUIIP)/simdispw.h $(GUIIP)/simctrlp.h $(GUIIP)/actdiagw.h $(GUIIP)/spikeratesdispw.h $(GUIIP)/simthread.h
 
UICOUT = $(GUIIP)/ui_mainw.h $(GUIIP)/ui_dispdialoguep.h $(GUIIP)/ui_conndispw.h $(GUIIP)/ui_simdispw.h $(GUIIP)/ui_simctrlp.h $(GUIIP)/ui_actdiagw.h $(GUIIP)/ui_spikeratesdispw.h
MOCOUT = $(GUIIP)/moc_mainw.h $(GUIIP)/moc_dispdialoguep.h $(GUIIP)/moc_conndispw.h $(GUIIP)/moc_simdispw.h $(GUIIP)/moc_simctrlp.h $(GUIIP)/moc_actdiagw.h $(GUIIP)/moc_spikeratesdispw.h $(GUIIP)/moc_simthread.h

GUIINC=$(MOCINC) $(UICOUT) $(MOCOUT)
BENCHINC=
EXPRINC=
COMINC = $(INCPATH)/common.h $(INCPATH)/parameters.h $(INCPATH)/globalvars.h $(INCPATH)/randomc.h $(INCPATH)/sfmt.h
SIMINC = $(INCPATH)/initsim.h $(INCPATH)/calcactivities.h
CUDAINC = $(CUDAIP)/kernels.h
 
MAININC= $(INCPATH)/main.h

MFMINC = $(MFMIP)/mfinputbase.h $(MFMIP)/mfinputec.h $(MFMIP)/mfinputcp.h
ERRMINC = $(ERRMIP)/errorinputbase.h $(ERRMIP)/errorinputec.h $(ERRMIP)/errorinputcp.h
OUTMINC = $(OUTMIP)/outputbase.h $(OUTMIP)/outputec.h $(OUTMIP)/outputcp.h
EXTMINC = $(EXTMIP)/cartpole.h $(EXTMIP)/externaldummy.h $(EXTMIP)/externalbase.h $(EXTERN)/cartpoleViz.h
MZMINC = $(MZMIP)/mzone.h
INMINC = $(INMIP)/innet.h $(INMIP)/innetnogo.h $(INMIP)/innetnogrgo.h $(INMIP)/innetnomfgo.h $(INMIP)/innetsparsegrgo.h
ANMINC = $(ANMIP)/psh.h $(ANMIP)/pshgpu.h

COREINC = $(MAININC) $(COMINC) $(SIMINC) $(CUDAINC) $(MFMINC) $(ERRMINC) $(OUTMINC) $(EXTMINC) $(MZMINC) $(INMINC) $(ANMINC)

GUIAPPINC = $(GUIINC) $(COREINC)
BENCHAPPINC=$(BENCHINC) $(COREINC)
EXPRAPPINC= $(EXPRINC) $(COREINC)

GUISRC = $(GUISP)/simthread.cpp $(GUISP)/mainw.cpp $(GUISP)/dispdialoguep.cpp $(GUISP)/conndispw.cpp $(GUISP)/simdispw.cpp $(GUISP)/simctrlp.cpp $(GUISP)/actdiagw.cpp $(GUISP)/spikeratesdispw.cpp
BENCHSRC = 
COMSRC = $(SRCPATH)/sfmt.cpp
SIMSRC = $(SRCPATH)/initsim.cpp $(SRCPATH)/calcactivities.cpp
CUDASRC = $(CUDASP)/kernels.cu
 
GMAINSRC = $(SRCPATH)/main.cpp
BMAINSRC = $(SRCPATH)/benchmain.cpp
EMAINSRC = $(SRCPATH)/exprmain.cpp

MFMSRC = $(MFMSP)/mfinputbase.cpp $(MFMSP)/mfinputec.cpp $(MFMSP)/mfinputcp.cpp
ERRMSRC = $(ERRMSP)/errorinputbase.cpp $(ERRMSP)/errorinputec.cpp $(ERRMSP)/errorinputcp.cpp
OUTMSRC = $(OUTMSP)/outputbase.cpp $(OUTMSP)/outputec.cpp $(OUTMSP)/outputcp.cpp
EXTMSRC = $(EXTMSP)/externalbase.cpp $(EXTMSP)/cartpole.cpp $(EXTMSP)/externaldummy.cpp $(EXTERN)/cartpoleViz.cpp
MZMSRC = $(MZMSP)/mzone.cpp
INMSRC = $(INMSP)/innet.cpp $(INMSP)/innetnogo.cpp $(INMSP)/innetnogrgo.cpp $(INMSP)/innetnomfgo.cpp $(INMSP)/innetsparsegrgo.cpp
ANMSRC =$(ANMSP)/psh.cpp $(ANMSP)/pshgpu.cpp

CORESRC = $(COMSRC) $(SIMSRC) $(CUDASRC) $(MFMSRC) $(ERRMSRC) $(OUTMSRC) $(EXTMSRC) $(MZMSRC) $(INMSRC) $(ANMSRC)

GUIAPPSRC=$(GMAINSRC) $(GUISRC) $(CORESRC)
BENCHAPPSRC=$(BMAINSRC) $(BENCHSRC) $(CORESRC)

GUIOBJ = $(OUTPATH)/mainw.obj $(OUTPATH)/dispdialoguep.obj $(OUTPATH)/conndispw.obj  $(OUTPATH)/simdispw.obj $(OUTPATH)/simctrlp.obj $(OUTPATH)/actdiagw.obj $(OUTPATH)/spikeratesdispw.obj $(OUTPATH)/simthread.obj
BENCHOBJ = 
EXPROBJ =
COMOBJ = $(OUTPATH)/sfmt.obj
SIMOBJ = $(OUTPATH)/initsim.obj $(OUTPATH)/calcactivities.obj
CUDAOBJ = $(OUTPATH)/kernels.obj
  
GMAINOBJ = $(OUTPATH)/main.obj
BMAINOBJ = $(OUTPATH)/benchmain.obj
EMAINOBJ = $(OUTPATH)/exprmain.obj

MFMOBJ = $(OUTPATH)/mfinputbase.obj $(OUTPATH)/mfinputec.obj $(OUTPATH)/mfinputcp.obj
ERRMOBJ = $(OUTPATH)/errorinputbase.obj $(OUTPATH)/errorinputec.obj $(OUTPATH)/errorinputcp.obj
OUTMOBJ = $(OUTPATH)/outputbase.obj $(OUTPATH)/outputec.obj $(OUTPATH)/outputcp.obj
EXTMOBJ = $(OUTPATH)/externalbase.obj $(OUTPATH)/cartpole.obj $(OUTPATH)/externaldummy.obj $(OUTPATH)/cartpoleViz.obj
MZMOBJ = $(OUTPATH)/mzone.obj
INMOBJ = $(OUTPATH)/innet.obj $(OUTPATH)/innetnogo.obj $(OUTPATH)/innetnogrgo.obj $(OUTPATH)/innetnomfgo.obj $(OUTPATH)/innetsparsegrgo.obj
ANMOBJ = $(OUTPATH)/psh.obj $(OUTPATH)/pshgpu.obj

COREOBJ = $(COMOBJ) $(SIMOBJ) $(CUDAOBJ) $(MFMOBJ) $(ERRMOBJ) $(OUTMOBJ) $(EXTMOBJ) $(MZMOBJ) $(INMOBJ) $(ANMOBJ)

GUIAPPOBJ = $(GMAINOBJ) $(GUIOBJ) $(COREOBJ)
BENCHAPPOBJ = $(BMAINOBJ) $(BENCHOBJ) $(COREOBJ)
EXPRAPPOBJ = $(EMAINOBJ) $(EXPROBJ) $(COREOBJ)

guiapp: guimain gui core
	-$(NVCC) $(GUIAPPOBJ) -o $(OUTPATH)/$(NAME) -L$(QTLIBPATH) -L$(CUDALIBPATH) -L$(INTELLIBPATH) $(QTLIBS) $(SDLLIBS) $(CUDALIBS) $(INTELLIBS)
guimain: $(GUIAPPINC) $(GMAINSRC)
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/main.cpp -o$(OUTPATH)/main.obj
gui: $(GUIAPPINC) $(GUISRC)
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(GUISP)/mainw.cpp -o$(OUTPATH)/mainw.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(GUISP)/dispdialoguep.cpp -o$(OUTPATH)/dispdialoguep.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(GUISP)/conndispw.cpp -o$(OUTPATH)/conndispw.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(GUISP)/simdispw.cpp -o$(OUTPATH)/simdispw.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(GUISP)/simctrlp.cpp -o$(OUTPATH)/simctrlp.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(GUISP)/actdiagw.cpp -o$(OUTPATH)/actdiagw.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(GUISP)/spikeratesdispw.cpp -o$(OUTPATH)/spikeratesdispw.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(GUISP)/simthread.cpp -o$(OUTPATH)/simthread.obj

guiinc: mocs uics


mocs: $(MOCINC)
	-$(MOC) $(GUIIP)/mainw.h -o $(GUIIP)/moc_mainw.h
	-$(MOC) $(GUIIP)/dispdialoguep.h -o $(GUIIP)/moc_dispdialoguep.h
	-$(MOC) $(GUIIP)/conndispw.h -o $(GUIIP)/moc_conndispw.h
	-$(MOC) $(GUIIP)/simdispw.h -o $(GUIIP)/moc_simdispw.h
	-$(MOC) $(GUIIP)/simctrlp.h -o $(GUIIP)/moc_simctrlp.h
	-$(MOC) $(GUIIP)/actdiagw.h -o $(GUIIP)/moc_actdiagw.h
	-$(MOC) $(GUIIP)/spikeratesdispw.h -o $(GUIIP)/moc_spikeratesdispw.h
	-$(MOC) $(GUIIP)/simthread.h -o $(GUIIP)/moc_simthread.h
uics: $(UIS)
	-$(UIC) $(GUIIP)/mainw.ui -o $(GUIIP)/ui_mainw.h
	-$(UIC) $(GUIIP)/dispdialoguep.ui -o $(GUIIP)/ui_dispdialoguep.h
	-$(UIC) $(GUIIP)/conndispw.ui -o $(GUIIP)/ui_conndispw.h
	-$(UIC) $(GUIIP)/simdispw.ui -o $(GUIIP)/ui_simdispw.h
	-$(UIC) $(GUIIP)/simctrlp.ui -o $(GUIIP)/ui_simctrlp.h
	-$(UIC) $(GUIIP)/actdiagw.ui -o $(GUIIP)/ui_actdiagw.h
	-$(UIC) $(GUIIP)/spikeratesdispw.ui -o $(GUIIP)/ui_spikeratesdispw.h


benchapp: benchmain bench core
	-$(NVCC) $(BENCHAPPOBJ) -o $(OUTPATH)/$(NAME)bench -L$(CUDALIBPATH) -L$(QTLIBPATH) -L$(INTELLIBPATH) $(QTLIBS) $(CUDALIBS) $(INTELLIBS)
benchmain: $(BENCHAPPINC) $(BMAINSRC)
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/benchmain.cpp -o$(OUTPATH)/benchmain.obj
bench:

exprapp: exprmain core
	-$(NVCC) $(EXPRAPPOBJ) -o $(OUTPATH)/$(NAME)expr -L$(CUDALIBPATH) -L$(QTLIBPATH) -L$(INTELLIBPATH) $(QTLIBS) $(SDLLIBS) $(CUDALIBS) $(INTELLIBS)
exprmain: $(EXPRAPPINC) $(EMAINSRC)
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/exprmain.cpp -o$(OUTPATH)/exprmain.obj


core: common sim cuda mfm errm outm extm mzm inm anm

common: $(COREINC) $(COMSRC)
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/sfmt.cpp -o$(OUTPATH)/sfmt.obj
sim: $(COREINC) $(SIMSRC)
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/initsim.cpp -o$(OUTPATH)/initsim.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/calcactivities.cpp -o$(OUTPATH)/calcactivities.obj

cuda: $(COREINC) $(CUDASRC)
	-$(NVCC) $(NVCFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(CUDASP)/kernels.cu -o $(OUTPATH)/kernels.obj

mfm: $(COREINC) $(MFMSRC)
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(MFMSP)/mfinputbase.cpp -o $(OUTPATH)/mfinputbase.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(MFMSP)/mfinputec.cpp -o $(OUTPATH)/mfinputec.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(MFMSP)/mfinputcp.cpp -o $(OUTPATH)/mfinputcp.obj
errm: $(COREINC) $(ERRMSRC)
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(ERRMSP)/errorinputbase.cpp -o $(OUTPATH)/errorinputbase.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(ERRMSP)/errorinputec.cpp -o $(OUTPATH)/errorinputec.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(ERRMSP)/errorinputcp.cpp -o $(OUTPATH)/errorinputcp.obj
outm: $(COREINC) $(OUTMSRC)
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(OUTMSP)/outputbase.cpp -o $(OUTPATH)/outputbase.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(OUTMSP)/outputec.cpp -o $(OUTPATH)/outputec.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(OUTMSP)/outputcp.cpp -o $(OUTPATH)/outputcp.obj	
extm: $(COREINC) $(EXTMSRC)
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(EXTMSP)/externalbase.cpp -o $(OUTPATH)/externalbase.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(EXTMSP)/cartpole.cpp -o $(OUTPATH)/cartpole.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(EXTERN)/cartpoleViz.cpp -o $(OUTPATH)/cartpoleViz.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(EXTMSP)/externaldummy.cpp -o $(OUTPATH)/externaldummy.obj
mzm: $(COREINC) $(MZMSRC)
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(MZMSP)/mzone.cpp -o $(OUTPATH)/mzone.obj
inm: $(COREINC) $(INMSRC)
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(INMSP)/innet.cpp -o $(OUTPATH)/innet.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(INMSP)/innetnogo.cpp -o $(OUTPATH)/innetnogo.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(INMSP)/innetnogrgo.cpp -o $(OUTPATH)/innetnogrgo.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(INMSP)/innetnomfgo.cpp -o $(OUTPATH)/innetnomfgo.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(INMSP)/innetsparsegrgo.cpp -o $(OUTPATH)/innetsparsegrgo.obj
anm: $(COREINC) $(ANMSRC)
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(ANMSP)/psh.cpp -o $(OUTPATH)/psh.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(ANMSP)/pshgpu.cpp -o $(OUTPATH)/pshgpu.obj

clean: cleangui cleanbench cleanexpr

cleangui: appcleangui fcleangui

appcleangui:
	-$(RM) $(OUTPATH)/$(NAME)
fcleangui:
	-$(RM) $(GUIAPPOBJ)

cleanbench: appcleanbench fcleanbench

appcleanbench:
	-$(RM) $(OUTPATH)/$(NAME)bench
fcleanbench:
	-$(RM) $(BENCHAPPOBJ)

cleanexpr: appcleanexpr fcleanexpr

appcleanexpr:
	-$(RM) $(OUTPATH)/$(NAME)expr
fcleanexpr:
	-$(RM) $(EXPRAPPOBJ)

rungui: $(OUTPATH)/$(NAME)
	-$(OUTPATH)/$(NAME) simstate1 psh1

runbench: $(OUTPATH)/$(NAME)bench
	-$(OUTPATH)/$(NAME)bench

runexpr: $(OUTPATH)/$(NAME)expr
	-$(OUTPATH)/$(NAME)expr
