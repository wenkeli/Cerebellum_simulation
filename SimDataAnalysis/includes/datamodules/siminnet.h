/*
 * siminnet.h
 *
 *  Created on: Aug 15, 2011
 *      Author: consciousness
 */

#ifndef SIMINNET_H_
#define SIMINNET_H_

#include <fstream>
#include <iostream>

#define MAXNGRDENPERGL 80

using namespace std;

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

class SimInNet
{
public:
	SimInNet(ifstream &infile);

	static const unsigned int numMF=1024;
	static const unsigned int numGO=1024;
	static const unsigned int numGR=1048576;
	static const unsigned int numSC=512;
	static const unsigned int numBC=128;
	static const unsigned int numGL=65536;

private:
	SimInNet();

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
	unsigned int apBufMF[numMF];

	//connectivity
	short numGROutPerMF[numMF];
	unsigned int mfConMFOutGR[numMF][maxNumGROutPerMF];

	char numGOOutPerMF[numMF];
	unsigned int mfConMFOutGO[numMF][maxNumGOOutPerMF];
	//---------end mossy fiber variables

	//---------golgi cell variables
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

	//---------end golgi cell variables

	//---------granule cell variables

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


	//---------end granule cell variables

	//--------stellate cell variables

	bool apSC[numSC];
	unsigned int apBufSC[numSC];

	float gPFSC[numSC];
	float threshSC[numSC];
	float vSC[numSC];

	//------------ end stellate cell variables
};

#endif /* SIMINNET_H_ */
