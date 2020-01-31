/*
 * mzone.h
 *
 *  Created on: Jun 13, 2011
 *      Author: consciousness
 */

#ifndef MZONE_H_
#define MZONE_H_

#include "../common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>

#include "../cuda/kernels.h"

class MZone
{
public:
	MZone(const bool *actSCIn, const bool *actMFIn, const bool *hMFIn,
			const unsigned int *pfBCSumIn, unsigned int *actBufGRGPU, unsigned int *delayMaskGRGPU, unsigned long *histGRGPU);
	MZone(ifstream &infile, const bool *actSCIn, const bool *actMFIn, const bool *hMFIn,
			const unsigned int *pfBCSumIn, unsigned int *actBufGRGPU, unsigned int *delayMaskGRGPU, unsigned long *histGRGPU);

	void exportState(ofstream &outfile);

	void assignBCOutPCCon();
	void assignPCOutNCCon();
	void assignIOCoupleCon();
	void cpyPFPCSynWCUDA();

	~MZone();

	void setErrDrive(float ed){errDrive=ed;};

	void runPFPCOutCUDA(cudaStream_t &st);
	void runPFPCSumCUDA(cudaStream_t &st);
	void cpyPFPCSumCUDA(cudaStream_t &st);
	void runPFPCPlastCUDA(cudaStream_t *sts, short t);

	void calcPCActivities();
	void calcBCActivities();
	void calcIOActivities();
	void calcNCActivities();

	void updatePCOut();
	void updateBCPCOut();
	void updateSCPCOut();
	void updateIOOut();
	void updateIOCouple();
	void updateNCOut();
	void updateMFNCOut();
	void updateMFNCSyn(short t);

	const bool *exportNCAct(){return (const bool *)apNC;};

	const unsigned int *exportApBufBC(){return (const unsigned int *)apBufBC;};
	const unsigned int *exportApBufPC(){return (const unsigned int *)apBufPC;};
	const unsigned int *exportApBufIO(){return (const unsigned int *)apBufIO;};
	const unsigned int *exportApBufNC(){return (const unsigned int *)apBufNC;};

	void exportActsPCBCDisp(SCBCPCActs &actSt);

	void exportActsIONCPCDisp(IONCPCActs &actSt);

        void disablePlasticity();
        void annealPlasticity(float decay);

	static const unsigned int numMF=1024;
	static const unsigned int numGR=1048576;
	static const unsigned int numSC=512;
	static const unsigned int numBC=128;
	static const unsigned int numPC=32;
	static const unsigned int numIO=4;
	static const unsigned int numNC=8;

private:
	MZone();

	void initCUDA(const unsigned int *pfBCSumIn, unsigned int *actBufGRGPU, unsigned int *delayMaskGRGPU, unsigned long *histGRGPU);

	//mossy fiber variables
	const bool *apMFInput;
	const bool *histMFInput;

	//stellate cell variables
	static const unsigned int numPCOutPerSC=1;
	const bool *apSCInput;

	//basket cell variables
	static const unsigned int numPCInperBC=4;
	static const unsigned int numPCOutPerBC=4;

	static const float eLeakBC;
	static const float ePCBC;
	static const float gLeakBC;
	static const float gPFDecayTBC;
	static const float gPFDecayBC;
	static const float gPCDecayTBC;
	static const float gPCDecayBC;
	static const float threshDecayTBC;
	static const float threshDecayBC;
	static const float threshBaseBC;
	static const float threshMaxBC;
	static const float pfIncConstBC;
	static const float pcIncConstBC;

	const unsigned int *sumPFBCInput;

	bool apBC[numBC];
	unsigned int apBufBC[numBC];

	unsigned char inputPCBC[numBC];
	float gPFBC[numBC];
	float gPCBC[numBC];
	float threshBC[numBC];
	float vBC[numBC];

	unsigned char bcConBCOutPC[numBC][numPCOutPerBC];

	//purkinje cell variables
	static const unsigned int numBCInPerPC=16;
	static const unsigned int numSCInPerPC=16;
	static const unsigned int numPFInPerPC=32768;
	static const unsigned int numBCOutPerPC=16;
	static const unsigned int bcToPCRatio=4;
	static const unsigned int numNCOutPerPC=3;

	static const float pfSynWInitPC;//=0.5;

	static const float eLeakPC;//=-60;
	static const float eBCPC;//=-80;
	static const float eSCPC;//=-80;
	static const float threshMaxPC;//=-48;
	static const float threshBasePC;//=-60;
	static const float threshDecayTPC;//=5;
	static const float threshDecayPC;//=1-exp(-TIMESTEP/threshDecayTPC);
	static const float gLeakPC;//=0.2/(6-TIMESTEP);
	static const float gPFDecayTPC;//=4.15;
	static const float gPFDecayPC;//=exp(-TIMESTEP/gPFDecayTPC);
	static const float gBCDecayTPC;//=5;
	static const float gBCDecayPC;//=exp(-TIMESTEP/gBCDecayTPC);
	static const float gSCDecayTPC;//=4.15;
	static const float gSCDecayPC;//=exp(-TIMESTEP/gSCDecayTPC);
	static const float gSCIncConstPC;//=0.0025;
	static const float gPFScaleConstPC;//=0.000065;
	static const float gBCScaleConstPC;//=0.037;

	static const unsigned int histBinWidthPC=5;
	static const unsigned int numHistBinsPC=8;

	float *pfSynWeightPCGPU;
	float pfSynWeightPCH[numGR];
	float *inputPFPCGPU;
	size_t inputPFPCGPUPitch;
	float *inputSumPFPCMZGPU;
	float *inputSumPFPCMZH;

	unsigned int *apBufGRGPU;
	unsigned int *delayBCPCSCMaskGRGPU;
	unsigned long *historyGRGPU;

	bool apPC[numPC];
	unsigned int apBufPC[numPC];

	unsigned char inputBCPC[numPC];
	bool inputSCPC[numPC][numSCInPerPC];
	float gPFPC[numPC];
	float gBCPC[numPC];
	float gSCPC[numPC][numSCInPerPC];
	float vPC[numPC];
	float threshPC[numPC];

//	unsigned char pcConPCtoBC[numPC][numBCOutPerPC];
	unsigned int pcConPCOutNC[numPC][numNCOutPerPC];

	//variables for MF-NC plasticity
	unsigned short histAllAPPC[numHistBinsPC];
	unsigned short histSumAllAPPC;
	unsigned char histBinNPC;
	short allAPPC;

	//IO cell variables
	static const unsigned int numIOCoupInPerIO=numIO-1;//1;
	static const unsigned int numNCInPerIO=8;
	static const float coupleScaleIO;//=0.04;

	static const float eLeakIO;//=-60;
	static const float eNCIO;//=-80;
	static const float gLeakIO;//=0.03;
	static const float gNCDecTSIO;//=0.5;
	static const float gNCDecTTIO;//=70;
	static const float gNCDecT0IO;//=0.56;
	static const float gNCIncScaleIO;//=0.003;
	static const float gNCIncTIO;//=300;
	static const float threshBaseIO;//=-61;
	static const float threshMaxIO;//=10;
	static const float threshTauIO;//=122;
	static const float threshDecayIO;//=1-exp(1-TIMESTEP/threshTauIO);

	static float pfPCLTPIncPF;//=0.0001;
	static float pfPCLTDDecPF;//=-0.001;
	static const unsigned int numHistBinsGR=40;
	static const unsigned int histBinWidthGR=5;
	static const int pfLTDTimerStartIO=-100;

	float errDrive;

	bool apIO[numIO];
	unsigned int apBufIO[numIO];

	bool inputNCIO[numIO][numNCInPerIO];
	float gNCIO[numIO][numNCInPerIO];
	float threshIO[numIO];
	float vIO[numIO];
	float vCoupIO[numIO];

	unsigned char conIOCouple[numIO][numIOCoupInPerIO];

	//plasticity variables
	int pfPlastTimerIO[numIO];

	//nucleus cell variables
	static const unsigned int numMFInPerNC=128;//1024;//128;//numMF;//128;
	static const unsigned int numPCInPerNC=12;

	static const float eLeakNC;//=-65;
	static const float ePCNC;//=-80;
	static const float mfNMDADecayTNC;//=50;
	static const float mfNMDADecayNC;//=exp(-TIMESTEP/mfNMDADecayTNC);
	static const float mfAMPADecayTNC;//=6;
	static const float mfAMPADecayNC;//=exp(-TIMESTEP/mfAMPADecayTNC);
	static const float gMFNMDAIncNC;//=1-exp(-TIMESTEP/3.0);
	static const float gMFAMPAIncNC;//=1-exp(-TIMESTEP/3.0);
	static const float gPCScaleAvgNC;//=0.177;
	static const float gPCDecayTNC;//=4.15;
	static const float gPCDecayNC;//=exp(-TIMESTEP/gPCDecayTNC);
	static const float gLeakNC;//=0.02;
	static const float threshDecayTNC;//=5;
	static const float threshDecayNC;//=1-exp(-TIMESTEP/threshDecayTNC);
	static const float threshMaxNC;//=-40;
	static const float threshBaseNC;//=-72;

	static const float outIORelPDecTSNC;//=40;
	static const float outIORelPDecTTNC;//=1;
	static const float outIORelPDecT0NC;//=78;
	static const float outIORelPIncScaleNC;//=0.25;
	static const float outIORelPIncTNC;//=0.8;

	static const float mfSynWInitNC;//=0.005;

	static const float mfNCLTDThresh;//=12;
	static const float mfNCLTPThresh;//=2;
	static float mfNCLTDDecNC;//=-0.0000025;
	static float mfNCLTPIncNC;//=0.0002;

	bool apNC[numNC];
	unsigned int apBufNC[numNC];

	bool inputPCNC[numNC][numPCInPerNC];
	float gPCNC[numNC][numPCInPerNC];
	float gPCScaleNC[numNC][numPCInPerNC];
	bool inputMFNC[numNC][numMFInPerNC];
	float mfSynWNC[numNC][numMFInPerNC];
	float mfNMDANC[numNC][numMFInPerNC];
	float mfAMPANC[numNC][numMFInPerNC];
	float gMFNMDANC[numNC][numMFInPerNC];
	float gMFAMPANC[numNC][numMFInPerNC];

	float threshNC[numNC];
	float vNC[numNC];

	float synIOPReleaseNC[numNC];

	bool noLTPMFNC;
	bool noLTDMFNC;
};

#endif /* MZONE_H_ */
