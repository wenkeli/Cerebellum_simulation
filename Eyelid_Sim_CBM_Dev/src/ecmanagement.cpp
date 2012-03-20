/*
 * ecmanagement.cpp
 *
 *  Created on: Mar 8, 2012
 *      Author: consciousness
 */

#include "../includes/ecmanagement.h"

using namespace std;

ECManagement::ECManagement(int numT, int iti)
{
	numTrials=numT;
	interTrialI=iti;

	cerr<<"numTrials: "<<numTrials<<" iti:"<<interTrialI<<endl;

	currentTrial=0;
	currentTime=0;

	simulation=new CBMSimCore(1);

	numMF=simulation->getNumMF();
	mf=new MFPoissonRegen(numMF, 1, 0.001);
	mfFreq=new float[numMF];

	for(int i=0; i<numMF; i++)
	{
		mfFreq[i]=10;
	}
}

ECManagement::~ECManagement()
{
	delete[] mfFreq;
	delete simulation;
	delete mf;
}

bool ECManagement::runStep()
{
	const bool *apMF;
	const bool *apGO;
	const bool *apGR;
	const float *vGR;
	const float *gESumGR;
	const float *gISumGR;

//	apMF=mf->getAPMF();
//	apGO=simulation->exportAPGO();
//	apGR=simulation->exportAPGR();
//	vGR=simulation->exportVmGR();
//	gESumGR=simulation->exportGESumGR();
//	gISumGR=simulation->exportGISumGR();

//	cout<<currentTrial<<" "<<currentTime<<" "<<
//			apMF[0]<<apMF[1]<<apMF[2]<<apMF[3]<<" "<<
//			apGO[0]<<apGO[1]<<apGO[2]<<apGO[3]<<" "<<
//			apGR[0]<<apGR[1]<<apGR[2]<<apGR[3]<<" "<<
//			vGR[0]<<"|"<<vGR[1]<<"|"<<vGR[2]<<"|"<<vGR[3]<<" "<<
//			gESumGR[0]<<"|"<<gESumGR[1]<<"|"<<gESumGR[2]<<"|"<<gESumGR[3]<<" "<<
//			gISumGR[0]<<"|"<<gISumGR[1]<<"|"<<gISumGR[2]<<"|"<<gISumGR[3]<<endl;

	if(currentTime>=interTrialI)
	{
		currentTime=0;
		currentTrial++;
	}
	if(currentTrial>=numTrials)
	{
		return false;
	}

	currentTime++;

	simulation->updateMFInput(mf->calcActivity(mfFreq));
	simulation->calcActivity();

	return true;
}

int ECManagement::getCurrentTrialN()
{
	return currentTrial;
}

int ECManagement::getCurrentTime()
{
	return currentTime;
}

const bool* ECManagement::exportAPMF()
{
	return mf->getAPMF();
}

const bool* ECManagement::exportAPGO()
{
	return simulation->exportAPGO();
}

const bool* ECManagement::exportAPGR()
{
	return simulation->exportAPGR();
}

const bool* ECManagement::exportAPSC()
{
	return simulation->exportAPSC();
}

const bool* ECManagement::exportAPBC()
{
	return simulation->exportAPBC(0);
}

const bool* ECManagement::exportAPPC()
{
	return simulation->exportAPPC(0);
}

const bool* ECManagement::exportAPIO()
{
	return simulation->exportAPIO(0);
}

const bool* ECManagement::exportAPNC()
{
	return simulation->exportAPNC(0);
}

const unsigned int* ECManagement::exportAPBufMF()
{
	return simulation->exportAPBufMF();
}

const unsigned int* ECManagement::exportAPBufGO()
{
	return simulation->exportAPBufGO();
}

const unsigned int* ECManagement::exportAPBufGR()
{
	return simulation->exportAPBufGR();
}

const unsigned int* ECManagement::exportAPBufSC()
{
	return simulation->exportAPBufSC();
}

const unsigned int* ECManagement::exportAPBufBC()
{
	return simulation->exportAPBufBC(0);
}

const unsigned int* ECManagement::exportAPBufPC()
{
	return simulation->exportAPBufPC(0);
}

const unsigned int* ECManagement::exportAPBufIO()
{
	return simulation->exportAPBufIO(0);
}

const unsigned int* ECManagement::exportAPBufNC()
{
	return simulation->exportAPBufNC(0);
}

unsigned int ECManagement::getGRX()
{
	return simulation->getGRX();
}

unsigned int ECManagement::getGRY()
{
	return simulation->getGRY();
}

unsigned int ECManagement::getGOX()
{
	return simulation->getGOX();
}

unsigned int ECManagement::getGOY()
{
	return simulation->getGOY();
}

unsigned int ECManagement::getGLX()
{
	return simulation->getGLX();
}

unsigned int ECManagement::getGLY()
{
	return simulation->getGLY();
}

unsigned int ECManagement::getNumMF()
{
	return simulation->getNumMF();
}

unsigned int ECManagement::getNumGO()
{
	return simulation->getNumGO();
}

unsigned int ECManagement::getNumGR()
{
	return simulation->getNumGR();
}

unsigned int ECManagement::getNumSC()
{
	return simulation->getNumSC();
}

unsigned int ECManagement::getNumBC()
{
	return simulation->getNumBC();
}

unsigned int ECManagement::getNumPC()
{
	return simulation->getNumPC();
}

unsigned int ECManagement::getNumNC()
{
	return simulation->getNumNC();
}

unsigned int ECManagement::getNumIO()
{
	return simulation->getNumIO();
}
