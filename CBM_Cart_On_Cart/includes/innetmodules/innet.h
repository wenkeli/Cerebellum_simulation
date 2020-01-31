/*
 * InNet.h
 *
 *  Created on: Jun 21, 2011
 *      Author: consciousness
 */

#ifndef INNET_H_
#define INNET_H_

#include "../common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>

#include "../cuda/kernels.h"

struct Glomerulus
{
	bool hasGODen;
	bool hasGOAx;
	bool hasMF;
	short goDenInd;
	short goAxInd;
	short mfInd;
	int grDenInds[MAXNGRDENPERGL];
	char numGRDen;
};

class InNet
{
public:
	InNet(const bool *actInMF);
	InNet(ifstream &infile, const bool *actInMF);
	virtual ~InNet();

	virtual void exportState(ofstream &outfile);

	virtual const bool *exportApSC(){return (const bool *)apSC;};
	virtual const bool *exportHistMF(){return (const bool *)histMF;};
	virtual const unsigned int *exportApBufMF(){return (const unsigned int *)apBufMF;};
	virtual const unsigned int *exportApBufGO(){return (const unsigned int *)apBufGO;};
	virtual const unsigned int *exportApBufSC(){return (const unsigned int *)apBufSC;};


	virtual void exportActSCDisp(SCBCPCActs &sbp);
	virtual void exportActGRDisp(vector<bool> &apRaster, int numCells);
	virtual void exportActGODisp(vector<bool> &apRaster, int numCells);

	virtual const unsigned int *exportPFBCSum(){return (const unsigned int *) inputSumPFBCH;};
	virtual unsigned int *exportApBufGRGPU(){return apBufGRGPU;};
	virtual unsigned int *exportDelayBCPCSCMaskGPU(){return delayBCPCSCMaskGRGPU;};
	virtual unsigned long *exportHistGRGPU(){return historyGRGPU;};

	virtual void updateMFActivties();
	virtual void calcGOActivities();
	virtual void calcSCActivities();
	virtual void updateMFtoGOOut();
	virtual void resetMFHist(short t);

	virtual void runGRActivitiesCUDA(cudaStream_t &st);
	virtual void runSumPFBCCUDA(cudaStream_t &st);
	virtual void runSumPFSCCUDA(cudaStream_t &st);
	virtual void runSumGRGOOutCUDA(cudaStream_t &st);
	virtual void cpyAPMFHosttoGPUCUDA(cudaStream_t &st);
	virtual void cpyAPGOHosttoGPUCUDA(cudaStream_t &st);
	virtual void runUpdateMFInGRCUDA(cudaStream_t &st);
	virtual void runUpdateGOInGRCUDA(cudaStream_t &st);
	virtual void runUpdatePFBCSCOutCUDA(cudaStream_t &st);
	virtual void runUpdateGROutGOCUDA(cudaStream_t &st);
	virtual void cpyPFBCSumGPUtoHostCUDA(cudaStream_t &st);
	virtual void cpyPFSCSumGPUtoHostCUDA(cudaStream_t &st);
	virtual void cpyGRGOSumGPUtoHostCUDA(cudaStream_t &st);
	virtual void runUpdateGRHistoryCUDA(cudaStream_t &st, short t);

	static const unsigned int numMF=1024;
	static const unsigned int numGO=1024;
	static const unsigned int numGR=1048576;
	static const unsigned int numSC=512;
	static const unsigned int numBC=128;
	static const unsigned int numGL=65536;

protected:
	virtual void initCUDA();
	virtual void initMFCUDA();
	virtual void initGRCUDA();
	virtual void initGOCUDA();
	virtual void initBCCUDA();
	virtual void initSCCUDA();

	virtual void connectNetwork();
	virtual void assignGRGL(stringstream &statusOut);
	virtual void assignGOGL(stringstream &statusOut);
	virtual void assignMFGL(stringstream &statusOut);
	virtual void translateMFGL(stringstream &statusOut);
	virtual void translateGOGL(stringstream &statusOut);
	virtual void assignGRGO(stringstream &statusOut, unsigned int nGRInPerGO);
	virtual void assignGRDelays(stringstream &statusOut);


	//connectivity definitions
	//glomeruli
	static const unsigned int maxNumGRDenPerGL=80;
	static const unsigned int normNumGRDenPerGL=64;
	static const unsigned int maxNumGODenPerGL=1;
	static const unsigned int maxNumGOAxPerGL=1;

	static const unsigned int glX=512;
	static const unsigned int glY=128;
	//end glomeruli

	//mossy fiber
	static const unsigned int numGLOutPerMF=64;
	static const unsigned int maxNumGOOutPerMF=64;
	static const unsigned int maxNumGROutPerMF=5120;
	//end mossy fibers

	//golgi cells
	static const unsigned int maxNumGRInPerGO=2048;
	static const unsigned int maxNumGLInPerGO=16;
	static const unsigned int maxNumMFInPerGO=16;
	static const unsigned int maxNumGLOutPerGO=48;
	static const unsigned int maxNumGROutPerGO=3840;

	static const unsigned int goX=64;
	static const unsigned int goY=16;

	static const unsigned int goGLDenSpanGLX=32;
	static const unsigned int goGLDenSpanGLY=32;

	static const unsigned int goPFDenSpanGRX=2048;
	static const unsigned int goPFDenSpanGRY=192;

	static const unsigned int goGLAxSpanGLX=32;
	static const unsigned int goGLAxSpanGLY=32;
	//end golgi cells

	//granule cells
	static const unsigned int maxNumGOOutPerGR=2;
	static const unsigned int maxNumInPerGR=4;
	static const unsigned int numPCOutPerPF=1;
	static const unsigned int numBCOutPerPF=1;
	static const unsigned int numSCOutPerPF=1;

	static const unsigned int grX=2048;
	static const unsigned int grY=512;

	static const unsigned int grGLDenSpanGLX=8;
	static const unsigned int grGLDenSpanGLY=8;
	//end granule cells

	//basket cells
	static const unsigned int numPFInPerBC=8192;
	//end basket cells

	//stellate cells
	static const unsigned int numPFInPerSC=2048;
	//end stellate cells

	//end connectivity definitions

	//---------glomeruli variables
	Glomerulus glomeruli[numGL];
	unsigned int grConGRInGL[numGR][maxNumInPerGR];
	unsigned int goConGOInGL[numGO][maxNumGLInPerGO];
	unsigned int goConGOOutGL[numGO][maxNumGLOutPerGO];
	unsigned int mfConMFOutGL[numMF][numGLOutPerMF];

	//---------end glomeruli variables

	//---------mossy fiber variables
	bool histMF[numMF];
	const bool *apMF;
	unsigned int apBufMF[numMF];

	//gpu related variables
	unsigned int *apMFH;
	unsigned int *apMFGPU;
	//end gpu related variables

	//connectivity
	short numGROutPerMF[numMF];
	unsigned int mfConMFOutGR[numMF][maxNumGROutPerMF];

	char numGOOutPerMF[numMF];
	unsigned int mfConMFOutGO[numMF][maxNumGOOutPerMF];
	//---------end mossy fiber variables

	//---------golgi cell variables
	float eLeakGO;
	float eMGluRGO;
	float threshMaxGO;
	float threshBaseGO;
	float gMFIncGO;
	float gGRIncGO;
	float gMGluRScaleGO;
	float gMGluRIncScaleGO;
	float mGluRScaleGO;
	float gluScaleGO;
	float gLeakGO;
	float gMFDecayTGO;
	float gMFDecayGO;
	float gGRDecayTGO;
	float gGRDecayGO;
	float mGluRDecayGO;
	float gMGluRIncDecayGO;
	float gMGluRDecayGO;
	float gluDecayGO;
	float threshDecayTGO;
	float threshDecayGO;

	bool apGO[numGO];
	unsigned int apBufGO[numGO];
	float vGO[numGO];
	float threshGO[numGO];
	unsigned short inputMFGO[numGO];
	float gMFGO[numGO];
	float gGRGO[numGO];
	float gMGluRGO[numGO];
	float gMGluRIncGO[numGO];
	float mGluRGO[numGO];
	float gluGO[numGO];

	//connectivity
	int numMFInPerGO[numGO];
	unsigned int goConMFOutGO[numGO][maxNumMFInPerGO];

	int numGROutPerGO[numGO];
	unsigned int goConGOOutGR[numGO][maxNumGROutPerGO];
	//end connectivity

	//gpu related variables
	unsigned int *apGOH;
	unsigned int *grInputGOSumH;

	unsigned int *apGOGPU;
	unsigned int *grInputGOGPU;
	size_t grInputGOGPUP;
	unsigned int *grInputGOSumGPU;
	//end gpu variables
	//---------end golgi cell variables

	//---------granule cell variables
	float eLeakGR;
	float eGOGR;
	float eMFGR;
	float threshMaxGR;
	float threshBaseGR;
	float gEIncGR;
	float gIIncGR;
	float gEDecayTGR;
	float gEDecayGR;
	float gIDecayTGR;
	float gIDecayGR;
	float threshDecayTGR;
	float threshDecayGR;
	float gLeakGR;

	static const unsigned int numHistBinsGR=40;
	static const unsigned int histBinWidthGR=5;


	//conduction delay
	unsigned int delayGOMasksGR[maxNumGOOutPerGR][numGR];
	unsigned int delayBCPCSCMaskGR[numGR];

	//connectivity
	int numGOOutPerGR[numGR];
	unsigned int grConGROutGO[maxNumGOOutPerGR][numGR];

	int numGOInPerGR[numGR];
	unsigned int grConGOOutGR[maxNumInPerGR][numGR];

	int numMFInPerGR[numGR];
	unsigned int grConMFOutGR[maxNumInPerGR][numGR];
	//end connectivity

	//gpu related variables
	//host variables
	unsigned char *outputGRH;
	//end host variables

	float *gEGRGPU;
	size_t gEGRGPUP;
	float *gEGRSumGPU;

	float *gIGRGPU;
	size_t gIGRGPUP;
	float *gIGRSumGPU;

	unsigned int *apBufGRGPU;
	float *threshGRGPU;
	float *vGRGPU;
	float *gKCaGRGPU;
	unsigned char *outputGRGPU;
	unsigned long *historyGRGPU;

	//conduction delays
	unsigned int *delayGOMasksGRGPU;
	size_t delayGOMasksGRGPUP;
	unsigned int *delayBCPCSCMaskGRGPU;

	//connectivity
	int *numGOOutPerGRGPU;
	unsigned int *grConGROutGOGPU;
	size_t grConGROutGOGPUP;

	int *numGOInPerGRGPU;
	unsigned int *grConGOOutGRGPU;
	size_t grConGOOutGRGPUP;

	int *numMFInPerGRGPU;
	unsigned int *grConMFOutGRGPU;
	size_t grConMFOutGRGPUP;
	//end gpu variables

	//---------end granule cell variables

	//--------stellate cell variables
	float eLeakSC;
	float gLeakSC;
	float gPFDecayTSC;
	float gPFDecaySC;
	float threshMaxSC;
	float threshBaseSC;
	float threshDecayTSC;
	float threshDecaySC;
	float pfIncSC;

	bool apSC[numSC];
	unsigned int apBufSC[numSC];

	float gPFSC[numSC];
	float threshSC[numSC];
	float vSC[numSC];

	//gpu related variables
	//host variables
	unsigned int *inputSumPFSCH;
	//end host variables

	unsigned int *inputPFSCGPU;
	size_t inputPFSCGPUP;
	unsigned int *inputSumPFSCGPU;
	//end gpu related variables

	//------------ end stellate cell variables

	//-----------basket cell variables
	//gpu related variables
	//host variables
	unsigned int *inputSumPFBCH;

	//device variables
	unsigned int *inputPFBCGPU;
	size_t inputPFBCGPUP;
	unsigned int *inputSumPFBCGPU;
	//end gpu related variables
	//-----------end basket cell variables

private:
	InNet();

};


#endif /* INNET_H_ */
