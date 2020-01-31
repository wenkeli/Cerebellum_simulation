#makes simulation -FIXED:no -Qparallel -Qopenmp_profile -Zi -debug:all -fixed:no -DDEBUG -DGPUDEBUG  -DACTDEBUG
NAME = cbm_new_CUDA

CC = icpc
NVCC = nvcc
DEFINES = -DINTELCC 
#-DDEBUG
CFLAGS = $(DEFINES) -O3
NVCFLAGS = -arch=compute_20 
#-code=sm_20
# -MD  -Qvc8 -Qms2 -Wall -Qvec-guard-write -Qopt-streaming-stores:always -Qopenmp  -Qipo -fast -MD  -O3

RM = rm
MOC = moc
UIC = uic

INTELLIBPATH ='/opt/intel/Compiler/11.0/072/lib/intel64/'
INTELLIBS = -lirc -lcxaguard

CUDAINCPATH = '/opt/cuda/include/'
CUDALIBPATH = '/opt/cuda/lib64/'
CUDALIBS = -lcuda 
#-lcudart

QTINCPATH = '/usr/include/qt4/'
QTLIBPATH = '/usr/lib/qt4/'
QTLIBS = -lQtGui -lQtCore

INCPATH	 = ./includes
SRCPATH = ./src
OUTPATH = ./output

UIS = $(INCPATH)/mainw.ui $(INCPATH)/dispdialoguep.ui $(INCPATH)/conndispw.ui $(INCPATH)/simdispw.ui $(INCPATH)/simctrlp.ui $(INCPATH)/actdiagw.ui $(INCPATH)/spikeratesdispw.ui
MOCINC = $(INCPATH)/mainw.h $(INCPATH)/dispdialoguep.h $(INCPATH)/conndispw.h $(INCPATH)/simdispw.h $(INCPATH)/simctrlp.h $(INCPATH)/actdiagw.h $(INCPATH)/spikeratesdispw.h $(INCPATH)/simthread.h
 
UICOUT = $(INCPATH)/ui_mainw.h $(INCPATH)/ui_dispdialoguep.h $(INCPATH)/ui_conndispw.h $(INCPATH)/ui_simdispw.h $(INCPATH)/ui_simctrlp.h $(INCPATH)/ui_actdiagw.h $(INCPATH)/ui_spikeratesdispw.h
MOCOUT = $(INCPATH)/moc_mainw.h $(INCPATH)/moc_dispdialoguep.h $(INCPATH)/moc_conndispw.h $(INCPATH)/moc_simdispw.h $(INCPATH)/moc_simctrlp.h $(INCPATH)/moc_actdiagw.h $(INCPATH)/moc_spikeratesdispw.h $(INCPATH)/moc_simthread.h

GUIINC=$(MOCINC) $(UICOUT) $(MOCOUT)
BENCHINC=
COMINC = $(INCPATH)/common.h $(INCPATH)/parameters.h $(INCPATH)/globalvars.h $(INCPATH)/randomc.h $(INCPATH)/sfmt.h
SIMINC = $(INCPATH)/synapsegenesis.h $(INCPATH)/initsim.h $(INCPATH)/calcactivities.h
CUDAINC = $(INCPATH)/commonCUDAKernels.h $(INCPATH)/grKernels.h $(INCPATH)/pcKernels.h $(INCPATH)/bcKernels.h $(INCPATH)/scKernels.h $(INCPATH)/ioKernels.h
MAININC=$(INCPATH)/main.h
IOINC = $(INCPATH)/writeout.h $(INCPATH)/readin.h
COREINC = $(MAININC) $(COMINC) $(SIMINC) $(CUDAINC) $(IOINC) 

GUIAPPINC = $(GUIINC) $(COREINC)
BENCHAPPINC=$(BENCHINC) $(COREINC)

GUISRC = $(SRCPATH)/simthread.cpp $(SRCPATH)/mainw.cpp $(SRCPATH)/dispdialoguep.cpp $(SRCPATH)/conndispw.cpp $(SRCPATH)/simdispw.cpp $(SRCPATH)/simctrlp.cpp $(SRCPATH)/actdiagw.cpp $(SRCPATH)/spikeratesdispw.cpp
BENCHSRC = 
COMSRC = $(SRCPATH)/sfmt.cpp
SIMSRC = $(SRCPATH)/synapsegenesis.cpp $(SRCPATH)/initsim.cpp $(SRCPATH)/calcactivities.cpp 
CUDASRC = $(SRCPATH)/commonCUDAKernels.cu $(SRCPATH)/grKernels.cu $(SRCPATH)/pcKernels.cu $(SRCPATH)/bcKernels.cu $(SRCPATH)/scKernels.cu $(SRCPATH)/ioKernels.cu
GMAINSRC = $(SRCPATH)/main.cpp
BMAINSRC = $(SRCPATH)/benchmain.cpp
IOSRC = $(SRCPATH)/writeout.cpp $(SRCPATH)/readin.cpp
CORESRC = $(COMSRC) $(SIMSRC) $(CUDASRC) $(IOSRC)

GUIAPPSRC=$(GMAINSRC) $(GUISRC) $(CORESRC)
BENCHAPPSRC=$(BMAINSRC) $(BENCHSRC) $(CORESRC)

GUIOBJ = $(OUTPATH)/mainw.obj $(OUTPATH)/dispdialoguep.obj $(OUTPATH)/conndispw.obj  $(OUTPATH)/simdispw.obj $(OUTPATH)/simctrlp.obj $(OUTPATH)/actdiagw.obj $(OUTPATH)/spikeratesdispw.obj $(OUTPATH)/simthread.obj
BENCHOBJ = 
COMOBJ = $(OUTPATH)/sfmt.obj
SIMOBJ = $(OUTPATH)/synapsegenesis.obj $(OUTPATH)/initsim.obj $(OUTPATH)/calcactivities.obj
CUDAOBJ = $(OUTPATH)/commonCUDAKernels.obj $(OUTPATH)/grKernels.obj $(OUTPATH)/pcKernels.obj $(OUTPATH)/bcKernels.obj $(OUTPATH)/scKernels.obj $(OUTPATH)/ioKernels.obj
GMAINOBJ = $(OUTPATH)/main.obj
BMAINOBJ = $(OUTPATH)/benchmain.obj
IOOBJ = $(OUTPATH)/writeout.obj $(OUTPATH)/readin.obj
COREOBJ = $(COMOBJ) $(SIMOBJ) $(CUDAOBJ) $(IOOBJ)

GUIAPPOBJ = $(GMAINOBJ) $(GUIOBJ) $(COREOBJ)
BENCHAPPOBJ = $(BMAINOBJ) $(BENCHOBJ) $(COREOBJ)

guiapp: guimain gui core
	-$(NVCC) $(GUIAPPOBJ) -o $(OUTPATH)/$(NAME) -L$(QTLIBPATH) -L$(CUDALIBPATH) -L$(INTELLIBPATH) $(QTLIBS) $(CUDALIBS) $(INTELLIBS)
guimain: $(GUIAPPINC) $(GMAINSRC)
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/main.cpp -o$(OUTPATH)/main.obj
gui: $(GUIAPPINC) $(GUISRC)
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/mainw.cpp -o$(OUTPATH)/mainw.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/dispdialoguep.cpp -o$(OUTPATH)/dispdialoguep.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/conndispw.cpp -o$(OUTPATH)/conndispw.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/simdispw.cpp -o$(OUTPATH)/simdispw.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/simctrlp.cpp -o$(OUTPATH)/simctrlp.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/actdiagw.cpp -o$(OUTPATH)/actdiagw.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/spikeratesdispw.cpp -o$(OUTPATH)/spikeratesdispw.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/simthread.cpp -o$(OUTPATH)/simthread.obj

guiinc: mocs uics
	

mocs: $(MOCINC)
	-$(MOC) $(INCPATH)/mainw.h -o $(INCPATH)/moc_mainw.h
	-$(MOC) $(INCPATH)/dispdialoguep.h -o $(INCPATH)/moc_dispdialoguep.h
	-$(MOC) $(INCPATH)/conndispw.h -o $(INCPATH)/moc_conndispw.h
	-$(MOC) $(INCPATH)/simdispw.h -o $(INCPATH)/moc_simdispw.h
	-$(MOC) $(INCPATH)/simctrlp.h -o $(INCPATH)/moc_simctrlp.h
	-$(MOC) $(INCPATH)/actdiagw.h -o $(INCPATH)/moc_actdiagw.h
	-$(MOC) $(INCPATH)/spikeratesdispw.h -o $(INCPATH)/moc_spikeratesdispw.h
	-$(MOC) $(INCPATH)/simthread.h -o $(INCPATH)/moc_simthread.h
uics: $(UIS)
	-$(UIC) $(INCPATH)/mainw.ui -o $(INCPATH)/ui_mainw.h
	-$(UIC) $(INCPATH)/dispdialoguep.ui -o $(INCPATH)/ui_dispdialoguep.h
	-$(UIC) $(INCPATH)/conndispw.ui -o $(INCPATH)/ui_conndispw.h
	-$(UIC) $(INCPATH)/simdispw.ui -o $(INCPATH)/ui_simdispw.h
	-$(UIC) $(INCPATH)/simctrlp.ui -o $(INCPATH)/ui_simctrlp.h
	-$(UIC) $(INCPATH)/actdiagw.ui -o $(INCPATH)/ui_actdiagw.h
	-$(UIC) $(INCPATH)/spikeratesdispw.ui -o $(INCPATH)/ui_spikeratesdispw.h


benchapp: benchmain bench core
	-$(NVCC) $(BENCHAPPOBJ) -o $(OUTPATH)/$(NAME)bench -L$(CUDALIBPATH) -L$(QTLIBPATH) -L$(INTELLIBPATH) $(QTLIBS) $(CUDALIBS) $(INTELLIBS)
benchmain: $(BENCHAPPINC) $(BMAINSRC)
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/benchmain.cpp -o$(OUTPATH)/benchmain.obj
bench:


core: common sim cuda io
	
common: $(COREINC) $(COMSRC)
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/sfmt.cpp -o$(OUTPATH)/sfmt.obj
sim: $(COREINC) $(SIMSRC)
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/synapsegenesis.cpp -o$(OUTPATH)/synapsegenesis.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/initsim.cpp -o$(OUTPATH)/initsim.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/calcactivities.cpp -o$(OUTPATH)/calcactivities.obj
cuda: $(COREINC) $(CUDASRC)
	-$(NVCC) $(NVCFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/commonCUDAKernels.cu -o $(OUTPATH)/commonCUDAKernels.obj
	-$(NVCC) $(NVCFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/grKernels.cu -o $(OUTPATH)/grKernels.obj
	-$(NVCC) $(NVCFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/pcKernels.cu -o $(OUTPATH)/pcKernels.obj
	-$(NVCC) $(NVCFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/bcKernels.cu -o $(OUTPATH)/bcKernels.obj
	-$(NVCC) $(NVCFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/scKernels.cu -o $(OUTPATH)/scKernels.obj
	-$(NVCC) $(NVCFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/ioKernels.cu -o $(OUTPATH)/ioKernels.obj	
io: $(COREINC) $(IOSRC)
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/writeout.cpp -o $(OUTPATH)/writeout.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/readin.cpp -o $(OUTPATH)/readin.obj
clean: cleangui cleanbench

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


rungui: $(OUTPATH)/$(NAME)
	-$(OUTPATH)/$(NAME) simstate1 psh1
#	-$(OUTPATH)/$(NAME) psh2
#	-$(OUTPATH)/$(NAME) psh3
#	-$(OUTPATH)/$(NAME) psh4
#	-$(OUTPATH)/$(NAME) psh5

runbench: $(OUTPATH)/$(NAME)bench
	-$(OUTPATH)/$(NAME)bench

