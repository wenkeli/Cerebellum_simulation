/*
 * ecmanagement.cpp
 *
 *  Created on: Mar 8, 2012
 *      Author: consciousness
 */

#include "../../includes/ectrial/ecmanagementbase.h"

using namespace std;

ECManagementBase::ECManagementBase(int numT, int iti)
{
	CRandomSFMT0 randGen(time(0));
	unsigned int numContextMFs;
	unsigned int *contextMFInds;

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
		mfFreq[i]=randGen.Random()*10;
	}

	numContextMFs=numMF*0.03;

	contextMFInds=new unsigned int[numContextMFs];

	for(int i=0; i<numContextMFs; i++)
	{
		while(true)
		{
			unsigned int newInd;
			bool indExist;

			indExist=false;

			newInd=randGen.IRandom(0, numMF);
			for(int j=0; j<i; j++)
			{
				if(contextMFInds[j]==newInd)
				{
					indExist=true;
				}
			}

			if(!indExist)
			{
				contextMFInds[i]=newInd;
				mfFreq[contextMFInds[i]]=randGen.Random()*30+30;
				break;
			}
		}
	}

	delete[] contextMFInds;
}

ECManagementBase::~ECManagementBase()
{
	delete[] mfFreq;
	delete simulation;
	delete mf;
}

bool ECManagementBase::runStep()
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

const bool* ECManagementBase::exportAPMF()
{
	return mf->getAPMF();
}

const bool* ECManagementBase::exportAPGO()
{
	return simulation->exportAPGO();
}

const bool* ECManagementBase::exportAPGR()
{
	return simulation->exportAPGR();
}

const bool* ECManagementBase::exportAPGL()
{
	return simulation->exportAPGL();
}

const bool* ECManagementBase::exportAPSC()
{
	return simulation->exportAPSC();
}

const bool* ECManagementBase::exportAPBC()
{
	return simulation->exportAPBC(0);
}

const bool* ECManagementBase::exportAPPC()
{
	return simulation->exportAPPC(0);
}

const bool* ECManagementBase::exportAPIO()
{
	return simulation->exportAPIO(0);
}

const bool* ECManagementBase::exportAPNC()
{
	return simulation->exportAPNC(0);
}

const float* ECManagementBase::exportVmGO()
{
	return simulation->exportVmGO();
}

const float* ECManagementBase::exportVmSC()
{
	return simulation->exportVmSC();
}

const float* ECManagementBase::exportVmBC()
{
	return simulation->exportVmBC(0);
}

const float* ECManagementBase::exportVmPC()
{
	return simulation->exportVmPC(0);
}

const float* ECManagementBase::exportVmIO()
{
	return simulation->exportVmIO(0);
}

const float* ECManagementBase::exportVmNC()
{
	return simulation->exportVmNC(0);
}

const unsigned int* ECManagementBase::exportAPBufMF()
{
	return simulation->exportAPBufMF();
}

const unsigned int* ECManagementBase::exportAPBufGO()
{
	return simulation->exportAPBufGO();
}

const unsigned int* ECManagementBase::exportAPBufGR()
{
	return simulation->exportAPBufGR();
}

const unsigned int* ECManagementBase::exportAPBufSC()
{
	return simulation->exportAPBufSC();
}

const unsigned int* ECManagementBase::exportAPBufBC()
{
	return simulation->exportAPBufBC(0);
}

const unsigned int* ECManagementBase::exportAPBufPC()
{
	return simulation->exportAPBufPC(0);
}

const unsigned int* ECManagementBase::exportAPBufIO()
{
	return simulation->exportAPBufIO(0);
}

const unsigned int* ECManagementBase::exportAPBufNC()
{
	return simulation->exportAPBufNC(0);
}

unsigned int ECManagementBase::getGRX()
{
	return simulation->getGRX();
}

unsigned int ECManagementBase::getGRY()
{
	return simulation->getGRY();
}

unsigned int ECManagementBase::getGOX()
{
	return simulation->getGOX();
}

unsigned int ECManagementBase::getGOY()
{
	return simulation->getGOY();
}

unsigned int ECManagementBase::getGLX()
{
	return simulation->getGLX();
}

unsigned int ECManagementBase::getGLY()
{
	return simulation->getGLY();
}

unsigned int ECManagementBase::getNumMF()
{
	return simulation->getNumMF();
}

unsigned int ECManagementBase::getNumGO()
{
	return simulation->getNumGO();
}

unsigned int ECManagementBase::getNumGR()
{
	return simulation->getNumGR();
}

unsigned int ECManagementBase::getNumGL()
{
	return simulation->getNumGL();
}

unsigned int ECManagementBase::getNumSC()
{
	return simulation->getNumSC();
}

unsigned int ECManagementBase::getNumBC()
{
	return simulation->getNumBC();
}

unsigned int ECManagementBase::getNumPC()
{
	return simulation->getNumPC();
}

unsigned int ECManagementBase::getNumNC()
{
	return simulation->getNumNC();
}

unsigned int ECManagementBase::getNumIO()
{
	return simulation->getNumIO();
}
