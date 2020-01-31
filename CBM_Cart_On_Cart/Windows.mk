#makes simulation -FIXED:no -Qparallel -Qopenmp_profile -Zi -debug:all -fixed:no -DDEBUG -DGPUDEBUG  -DACTDEBUG
NAME = cbm_new_CUDA

CC = icl
NVCC = nvcc
DEFINES = -DINTELCC 
#-DDEBUG
CFLAGS = $(DEFINES) -nologo -O3 -Oa -F100000000 -arch:SSE4.1 -MD
NVCFLAGS = -arch=compute_20
#-code=sm_20
# -MD  -Qvc8 -Qms2 -Wall -Qvec-guard-write -Qopt-streaming-stores:always -Qopenmp  -Qipo -fast -MD  -O3

RM = rm
MOC = moc
UIC = uic

CUDAINCPATH= 'C:/CUDA/include/'
CUDALIBPATH= 'C:/CUDA/lib/'
CUDALIBS= -DEFAULTLIB:cuda

QTINCPATH = 'C:/qt/4.6.3/include/'
QTLIBPATH = 'C:/qt/4.6.3/lib/'
QTLIBS = -DEFAULTLIB:qtmain,-DEFAULTLIB:QTGui4,-DEFAULTLIB:QTCore4

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
COREINC = $(MAININC) $(COMINC) $(SIMINC) $(CUDAINC) 

GUIAPPINC = $(GUIINC) $(COREINC)
BENCHAPPINC=$(BENCHINC) $(COREINC)

GUISRC = $(SRCPATH)/simthread.cpp $(SRCPATH)/mainw.cpp $(SRCPATH)/dispdialoguep.cpp $(SRCPATH)/conndispw.cpp $(SRCPATH)/simdispw.cpp $(SRCPATH)/simctrlp.cpp $(SRCPATH)/actdiagw.cpp $(SRCPATH)/spikeratesdispw.cpp
BENCHSRC = 
COMSRC = $(SRCPATH)/sfmt.cpp
SIMSRC = $(SRCPATH)/synapsegenesis.cpp $(SRCPATH)/initsim.cpp $(SRCPATH)/calcactivities.cpp 
CUDASRC = $(SRCPATH)/commonCUDAKernels.cu $(SRCPATH)/grKernels.cu $(SRCPATH)/pcKernels.cu $(SRCPATH)/bcKernels.cu $(SRCPATH)/scKernels.cu $(SRCPATH)/ioKernels.cu
GMAINSRC = $(SRCPATH)/main.cpp
BMAINSRC = $(SRCPATH)/benchmain.cpp
CORESRC = $(COMSRC) $(SIMSRC) $(CUDASRC)

GUIAPPSRC=$(GMAINSRC) $(GUISRC) $(CORESRC)
BENCHAPPSRC=$(BMAINSRC) $(BENCHSRC) $(CORESRC)

GUIOBJ = $(OUTPATH)/mainw.obj $(OUTPATH)/dispdialoguep.obj $(OUTPATH)/conndispw.obj  $(OUTPATH)/simdispw.obj $(OUTPATH)/simctrlp.obj $(OUTPATH)/actdiagw.obj $(OUTPATH)/spikeratesdispw.obj $(OUTPATH)/simthread.obj
BENCHOBJ = 
COMOBJ = $(OUTPATH)/sfmt.obj
SIMOBJ = $(OUTPATH)/synapsegenesis.obj $(OUTPATH)/initsim.obj $(OUTPATH)/calcactivities.obj
CUDAOBJ = $(OUTPATH)/commonCUDAKernels.obj $(OUTPATH)/grKernels.obj $(OUTPATH)/pcKernels.obj $(OUTPATH)/bcKernels.obj $(OUTPATH)/scKernels.obj $(OUTPATH)/ioKernels.obj
GMAINOBJ = $(OUTPATH)/main.obj
BMAINOBJ = $(OUTPATH)/benchmain.obj
COREOBJ = $(COMOBJ) $(SIMOBJ) $(CUDAOBJ)

GUIAPPOBJ = $(GMAINOBJ) $(GUIOBJ) $(COREOBJ)
BENCHAPPOBJ = $(BMAINOBJ) $(BENCHOBJ) $(COREOBJ)

guiapp: guimain gui core
	-$(NVCC) $(GUIAPPOBJ) -o $(OUTPATH)/$(NAME) -Xlinker -stack:100000000,-RELEASE,-Manifest,-LIBPATH:$(QTLIBPATH),$(QTLIBS),-LIBPATH:$(CUDALIBPATH),$(CUDALIBS),-NODEFAULTLIB:LIBCMT
guimain: mocs uics $(GUIAPPINC) $(GMAINSRC)
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/main.cpp -Fo$(OUTPATH)/main.obj
gui: mocs uics $(GUIAPPINC) $(GUISRC)
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/mainw.cpp -Fo$(OUTPATH)/mainw.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/dispdialoguep.cpp -Fo$(OUTPATH)/dispdialoguep.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/conndispw.cpp -Fo$(OUTPATH)/conndispw.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/simdispw.cpp -Fo$(OUTPATH)/simdispw.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/simctrlp.cpp -Fo$(OUTPATH)/simctrlp.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/actdiagw.cpp -Fo$(OUTPATH)/actdiagw.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/spikeratesdispw.cpp -Fo$(OUTPATH)/spikeratesdispw.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/simthread.cpp -Fo$(OUTPATH)/simthread.obj
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
	-$(NVCC) $(BENCHAPPOBJ) -o $(OUTPATH)/$(NAME)bench -Xlinker -stack:100000000,-RELEASE,-Manifest,-LIBPATH:$(QTLIBPATH),$(QTLIBS),-LIBPATH:$(CUDALIBPATH),$(CUDALIBS),-NODEFAULTLIB:LIBCMT
benchmain: $(BENCHAPPINC) $(BMAINSRC)
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/benchmain.cpp -Fo$(OUTPATH)/benchmain.obj
bench:


core: common sim cuda
	
common: $(COREINC) $(COMSRC)
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/sfmt.cpp -Fo$(OUTPATH)/sfmt.obj
sim: $(COREINC) $(SIMSRC)
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/synapsegenesis.cpp -Fo$(OUTPATH)/synapsegenesis.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/initsim.cpp -Fo$(OUTPATH)/initsim.obj
	-$(CC) $(CFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/calcactivities.cpp -Fo$(OUTPATH)/calcactivities.obj
cuda: $(COREINC) $(CUDASRC)
	-$(NVCC) $(NVCFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/commonCUDAKernels.cu -o $(OUTPATH)/commonCUDAKernels.obj
	-$(NVCC) $(NVCFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/grKernels.cu -o $(OUTPATH)/grKernels.obj
	-$(NVCC) $(NVCFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/pcKernels.cu -o $(OUTPATH)/pcKernels.obj
	-$(NVCC) $(NVCFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/bcKernels.cu -o $(OUTPATH)/bcKernels.obj
	-$(NVCC) $(NVCFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/scKernels.cu -o $(OUTPATH)/scKernels.obj
	-$(NVCC) $(NVCFLAGS) -I $(QTINCPATH) -I $(CUDAINCPATH) -c $(SRCPATH)/ioKernels.cu -o $(OUTPATH)/ioKernels.obj	


clean: cleangui cleanbench

cleangui: appcleangui fcleangui
	
appcleangui:
	-$(RM) $(OUTPATH)/$(NAME).exe $(OUTPATH)/$(NAME).exe.manifest
fcleangui:
	-$(RM) $(GUIAPPOBJ) $(UICOUT) $(MOCOUT)
	
cleanbench: appcleanbench fcleanbench
	
appcleanbench:
	-$(RM) $(OUTPATH)/$(NAME)bench.exe $(OUTPATH)/$(NAME)bench.exe.manifest
fcleanbench:
	-$(RM) $(BENCHAPPOBJ)


rungui: $(OUTPATH)/$(NAME).exe
	-$(OUTPATH)/$(NAME) psh1
#	-$(OUTPATH)/$(NAME) psh2
#	-$(OUTPATH)/$(NAME) psh3
#	-$(OUTPATH)/$(NAME) psh4
#	-$(OUTPATH)/$(NAME) psh5

runbench: $(OUTPATH)/$(NAME)bench.exe
	-$(OUTPATH)/$(NAME)bench.exe

