/*
 * ecmanagement.cpp
 *
 *  Created on: Mar 8, 2012
 *      Author: consciousness
 */

#include "../../includes/ectrial/ecmanagementbase.h"

using namespace std;

ECManagementBase::ECManagementBase(string conParamFile, string actParamFile,
		int numT, int iti, int randSeed, int gpuIndStart, int numGPUP2)
{
	fstream conPF;
	fstream actPF;

	conPF.open(conParamFile.c_str(), ios::in);
	actPF.open(actParamFile.c_str(), ios::in);

	simState=new CBMState(actPF, conPF, 1, randSeed, &randSeed, &randSeed);

	conPF.close();
	actPF.close();

	initialize(randSeed, numT, iti, gpuIndStart, numGPUP2);
}

ECManagementBase::ECManagementBase(string stateDataFile,
		int numT, int iti, int randSeed, int gpuIndStart, int numGPUP2)
{
	fstream stateDataF;

	stateDataF.open(stateDataFile.c_str(), ios::in|ios::binary);

	simState=new CBMState(stateDataF);

	stateDataF.close();

	initialize(randSeed, numT, iti, gpuIndStart, numGPUP2);
}

void ECManagementBase::initialize(int randSeed, int numT, int iti, int gpuIndStart, int numGPUP2)
{
	simulation=new CBMSimCore(simState, &randSeed, gpuIndStart, numGPUP2);

	simulation->writeToState();

	numMF=simState->getConnectivityParams()->getNumMF();
	numTrials=numT;
	interTrialI=iti;

	cerr<<"numTrials: "<<numTrials<<" iti:"<<interTrialI<<endl;

	currentTrial=0;
	currentTime=-1;
}

ECManagementBase::~ECManagementBase()
{
	delete simulation;
	delete simState;
}

bool ECManagementBase::runStep()
{
//	const bool *apMF;
//	const bool *apGO;
//	const bool *apGR;
//	const float *vGR;
//	const float *gESumGR;
//	const float *gISumGR;

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

	currentTime++;

	if(currentTime>=interTrialI)
	{
		currentTime=0;
		currentTrial++;
	}
	if(currentTrial>=numTrials)
	{
		return false;
	}

	calcMFActivity();

	calcSimActivity();

	return true;
}

int ECManagementBase::getCurrentTrialN()
{
	return currentTrial;
}

int ECManagementBase::getCurrentTime()
{
	return currentTime;
}

int ECManagementBase::getNumTrials()
{
	return numTrials;
}

int ECManagementBase::getInterTrialI()
{
	return interTrialI;
}

IConnectivityParams* ECManagementBase::getConParams()
{
	return simState->getConnectivityParams();
}

InNetInterface* ECManagementBase::getInputNet()
{
	return simulation->getInputNet();
}

MZoneInterface* ECManagementBase::getMZone()
{
	return simulation->getMZoneList()[0];
}

const ct_uint8_t* ECManagementBase::exportAPMF()
{
	return apMF;
}

void ECManagementBase::calcSimActivity()
{
	simulation->updateMFInput(apMF);
	simulation->calcActivity();
}

