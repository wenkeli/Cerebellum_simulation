/*
 * innetactivitystate.cpp
 *
 *  Created on: Nov 16, 2012
 *      Author: consciousness
 */

#include "../../CBMStateInclude/state/innetactivitystate.h"

using namespace std;

InNetActivityState::InNetActivityState
	(ConnectivityParams *conParams, ActivityParams *actParams)
{
	cp=conParams;

	allocateMemory();
	initializeVals(actParams);
}

InNetActivityState::InNetActivityState(ConnectivityParams *conParams, fstream &infile)
{
	cp=conParams;

	allocateMemory();

	stateRW(true, infile);
}

InNetActivityState::InNetActivityState(const InNetActivityState &state)
{
	cp=state.cp;

	allocateMemory();

	for(int i=0; i<cp->numMF; i++)
	{
		histMF[i]=state.histMF[i];
		apBufMF[i]=state.apBufMF[i];
	}

	for(int i=0; i<cp->numGO; i++)
	{
		apGO[i]=state.apGO[i];
		vGO[i]=state.vGO[i];
		vCoupleGO[i]=state.vCoupleGO[i];
		threshCurGO[i]=state.threshCurGO[i];
		inputMFGO[i]=state.inputMFGO[i];
		inputGOGO[i]=state.inputGOGO[i];
		gMFGO[i]=state.gMFGO[i];
		gGRGO[i]=state.gGRGO[i];
		gGOGO[i]=state.gGOGO[i];
		gMGluRGO[i]=state.gMGluRGO[i];
		gMGluRIncGO[i]=state.gMGluRIncGO[i];
		mGluRGO[i]=state.mGluRGO[i];
		gluGO[i]=state.gluGO[i];
	}

	for(int i=0; i<cp->numGR; i++)
	{
		apGR[i]=state.apGR[i];
		apBufGR[i]=state.apBufGR[i];
		for(int j=0; j<cp->maxnumpGRfromMFtoGR; j++)
		{
			gMFGR[i][j]=state.gMFGR[i][j];
		}
		gMFSumGR[i]=0;
		for(int j=0; j<cp->maxnumpGRfromGOtoGR; j++)
		{
			gGOGR[i][j]=state.gGOGR[i][j];
		}
		gGOSumGR[i]=state.gGOSumGR[i];
		threshGR[i]=state.threshGR[i];
		vGR[i]=state.vGR[i];
		gKCaGR[i]=state.gKCaGR[i];
		historyGR[i]=state.historyGR[i];
	}

	for(int i=0; i<cp->numSC; i++)
	{
		apSC[i]=state.apSC[i];
		apBufSC[i]=state.apBufSC[i];
		gPFSC[i]=state.gPFSC[i];
		threshSC[i]=state.threshSC[i];
		vSC[i]=state.vSC[i];
		inputSumPFSC[i]=state.inputSumPFSC[i];
	}

}

InNetActivityState::~InNetActivityState()
{
	delete[] histMF;
	delete[] apBufMF;

	delete[] apGO;
	delete[] apBufGO;
	delete[] vGO;
	delete[] vCoupleGO;
	delete[] threshCurGO;
	delete[] inputMFGO;
	delete[] inputGOGO;
	delete[] gMFGO;
	delete[] gGRGO;
	delete[] gGOGO;
	delete[] gMGluRGO;
	delete[] gMGluRIncGO;
	delete[] mGluRGO;
	delete[] gluGO;

	delete[] apGR;
	delete[] apBufGR;
	delete2DArray<float>(gMFGR);
	delete[] gMFSumGR;
	delete2DArray<float>(gGOGR);
	delete[] gGOSumGR;
	delete[] threshGR;
	delete[] vGR;
	delete[] gKCaGR;
	delete[] historyGR;

	delete[] apSC;
	delete[] apBufSC;
	delete[] gPFSC;
	delete[] threshSC;
	delete[] vSC;
	delete[] inputSumPFSC;
}

void InNetActivityState::writeState(fstream &outfile)
{
	stateRW(false, (fstream &)outfile);
}

bool InNetActivityState::equivalent(const InNetActivityState &compState)
{
	bool equal;

	equal=true;

	for(int i=0; i<cp->numMF; i++)
	{
		equal=equal && (histMF[i]==compState.histMF[i]);
		equal=equal && (apBufMF[i]==compState.apBufMF[i]);
	}

	for(int i=0; i<cp->numGO; i++)
	{
		equal=equal && (apGO[i]==compState.apGO[i]);
		equal=equal && (apBufGO[i]==compState.apBufGO[i]);
		equal=equal && (vGO[i]==compState.vGO[i]);
		equal=equal && (threshCurGO[i]==compState.threshCurGO[i]);
	}

	for(int i=0; i<cp->numSC; i++)
	{
		equal=equal && (vSC[i]==compState.vSC[i]);
	}

	return equal;
}

bool InNetActivityState::validateState()
{
	bool valid;

	valid=true;

	valid=valid && validateFloatArray(vGO, cp->numGO);
	valid=valid && validateFloatArray(gMFGO, cp->numGO);
	valid=valid && validateFloatArray(gGRGO, cp->numGO);

	valid=valid && validate2DfloatArray(gMFGR, cp->numGR*cp->maxnumpGRfromMFtoGR);
	valid=valid && validate2DfloatArray(gGOGR, cp->numGR*cp->maxnumpGRfromGOtoGR);
	valid=valid && validateFloatArray(vGR, cp->numGR);
	valid=valid && validateFloatArray(gKCaGR, cp->numGR);

	valid=valid && validateFloatArray(gPFSC, cp->numSC);
	valid=valid && validateFloatArray(vSC, cp->numSC);

	return valid;
}

void InNetActivityState::resetState(ActivityParams *ap)
{
	initializeVals(ap);
}

void InNetActivityState::allocateMemory()
{
	histMF=new ct_uint8_t[cp->numMF];
	apBufMF=new ct_uint32_t[cp->numMF];

	apGO=new ct_uint8_t[cp->numGO];
	apBufGO=new ct_uint32_t[cp->numGO];
	vGO=new float[cp->numGO];
	vCoupleGO=new float[cp->numGO];
	threshCurGO=new float[cp->numGO];
	inputMFGO=new ct_uint32_t[cp->numGO];
	inputGOGO=new ct_uint32_t[cp->numGO];

	//todo: synaptic depression test
	inputGOGABASynDepGO=new float[cp->numGO];
	goGABAOutSynScaleGOGO=new float[cp->numGO];

	gMFGO=new float[cp->numGO];
	gGRGO=new float[cp->numGO];
	gGOGO=new float[cp->numGO];
	gMGluRGO=new float[cp->numGO];
	gMGluRIncGO=new float[cp->numGO];
	mGluRGO=new float[cp->numGO];
	gluGO=new float[cp->numGO];

	apGR=new ct_uint8_t[cp->numGR];
	apBufGR= new ct_uint32_t[cp->numGR];
	gMFGR=allocate2DArray<float>(cp->numGR, cp->maxnumpGRfromMFtoGR);
	gMFSumGR=new float[cp->numGR];
	gGOGR=allocate2DArray<float>(cp->numGR, cp->maxnumpGRfromGOtoGR);
	gGOSumGR=new float[cp->numGR];
	threshGR=new float[cp->numGR];
	vGR=new float[cp->numGR];
	gKCaGR=new float[cp->numGR];
	historyGR=new ct_uint64_t[cp->numGR];

	apSC=new ct_uint8_t[cp->numSC];
	apBufSC=new ct_uint32_t[cp->numSC];
	gPFSC=new float[cp->numSC];
	threshSC=new float[cp->numSC];
	vSC=new float[cp->numSC];
	inputSumPFSC=new ct_uint32_t[cp->numSC];
}

void InNetActivityState::stateRW(bool read, fstream &file)
{
	rawBytesRW((char *)histMF, cp->numMF*sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufMF, cp->numMF*sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)apGO, cp->numGO*sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufGO, cp->numGO*sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)vGO, cp->numGO*sizeof(float), read, file);
	rawBytesRW((char *)vCoupleGO, cp->numGO*sizeof(float), read, file);
	rawBytesRW((char *)threshCurGO, cp->numGO*sizeof(float), read, file);
	rawBytesRW((char *)inputMFGO, cp->numGO*sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)inputGOGO, cp->numGO*sizeof(ct_uint32_t), read, file);

	//todo: synaptic depression test
	arrayInitialize<float>(inputGOGABASynDepGO, 1, cp->numGO);
	arrayInitialize<float>(goGABAOutSynScaleGOGO, 0, cp->numGO);

	rawBytesRW((char *)gMFGO, cp->numGO*sizeof(float), read, file);
	rawBytesRW((char *)gGRGO, cp->numGO*sizeof(float), read, file);
	rawBytesRW((char *)gGOGO, cp->numGO*sizeof(float), read, file);
	rawBytesRW((char *)gMGluRGO, cp->numGO*sizeof(float), read, file);
	rawBytesRW((char *)gMGluRIncGO, cp->numGO*sizeof(float), read, file);
	rawBytesRW((char *)mGluRGO, cp->numGO*sizeof(float), read, file);
	rawBytesRW((char *)gluGO, cp->numGO*sizeof(float), read, file);

	rawBytesRW((char *)apGR, cp->numGR*sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufGR, cp->numGR*sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)gMFGR[0], cp->numGR*cp->maxnumpGRfromMFtoGR*sizeof(float), read, file);
	rawBytesRW((char *)gMFSumGR, cp->numGR*sizeof(float), read, file);
	rawBytesRW((char *)gGOGR[0], cp->numGR*cp->maxnumpGRfromGOtoGR*sizeof(float), read, file);
	rawBytesRW((char *)gGOSumGR, cp->numGR*sizeof(float), read, file);
	rawBytesRW((char *)threshGR, cp->numGR*sizeof(float), read, file);
	rawBytesRW((char *)vGR, cp->numGR*sizeof(float), read, file);
	rawBytesRW((char *)gKCaGR, cp->numGR*sizeof(float), read, file);
	rawBytesRW((char *)historyGR, cp->numGR*sizeof(ct_uint64_t), read, file);

	rawBytesRW((char *)apSC, cp->numSC*sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufSC, cp->numSC*sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)gPFSC, cp->numSC*sizeof(float), read, file);
	rawBytesRW((char *)threshSC, cp->numSC*sizeof(float), read, file);
	rawBytesRW((char *)vSC, cp->numSC*sizeof(float), read, file);
	rawBytesRW((char *)inputSumPFSC, cp->numSC*sizeof(ct_uint32_t), read, file);
}

void InNetActivityState::initializeVals(ActivityParams *ap)
{
	for(int i=0; i<cp->numMF; i++)
	{
		histMF[i]=false;
		apBufMF[i]=0;
	}

	for(int i=0; i<cp->numGO; i++)
	{
		apGO[i]=false;
		apBufGO[i]=0;
		vGO[i]=ap->eLeakGO;
		vCoupleGO[i]=0;
		threshCurGO[i]=ap->threshRestGO;
		inputMFGO[i]=0;
		inputGOGO[i]=0;

		//todo: synaptic depression test
		inputGOGABASynDepGO[i]=0;
		goGABAOutSynScaleGOGO[i]=1;

		gMFGO[i]=0;
		gGRGO[i]=0;
		gGOGO[i]=0;
		gMGluRGO[i]=0;
		gMGluRIncGO[i]=0;
		mGluRGO[i]=0;
		gluGO[i]=0;
	}

	for(int i=0; i<cp->numGR; i++)
	{
		apGR[i]=false;
		apBufGR[i]=0;
		for(int j=0; j<cp->maxnumpGRfromMFtoGR; j++)
		{
			gMFGR[i][j]=0;
		}
		gMFSumGR[i]=0;
		for(int j=0; j<cp->maxnumpGRfromGOtoGR; j++)
		{
			gGOGR[i][j]=0;
		}
		gGOSumGR[i]=0;
		threshGR[i]=ap->threshRestGR;
		vGR[i]=ap->eLeakGR;
		gKCaGR[i]=0;
		historyGR[i]=0;
	}

	for(int i=0; i<cp->numSC; i++)
	{
		apSC[i]=false;
		apBufSC[i]=0;
		gPFSC[i]=0;
		threshSC[i]=ap->threshRestSC;
		vSC[i]=ap->eLeakSC;
		inputSumPFSC[i]=0;
	}
}
