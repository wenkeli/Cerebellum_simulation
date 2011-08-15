/*
 * simmzone.h
 *
 *  Created on: Aug 15, 2011
 *      Author: consciousness
 */

#ifndef SIMMZONE_H_
#define SIMMZONE_H_

#include <fstream>
#include <iostream>

using namespace std;

class SimMZone
{
public:
	SimMZone(ifstream &infile);

	static const unsigned int numMF=1024;
	static const unsigned int numGR=1048576;
	static const unsigned int numSC=512;
	static const unsigned int numBC=128;
	static const unsigned int numPC=32;
	static const unsigned int numIO=4;
	static const unsigned int numNC=8;
private:
	SimMZone();

	//stellate cell variables
	static const unsigned int numPCOutPerSC=1;

	//basket cell variables
	static const unsigned int numPCInperBC=4;
	static const unsigned int numPCOutPerBC=4;

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

	static const unsigned int histBinWidthPC=5;
	static const unsigned int numHistBinsPC=8;

	float pfSynWeightPCH[numGR];

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

#endif /* SIMMZONE_H_ */
