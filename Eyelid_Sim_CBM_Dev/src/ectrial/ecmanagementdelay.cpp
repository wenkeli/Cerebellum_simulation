/*
 * ecmanagementdelay.cpp
 *
 *  Created on: Sep 4, 2012
 *      Author: consciousness
 */

#include "../../includes/ectrial/ecmanagementdelay.h"

ECManagementDelay::ECManagementDelay(int numT, int iti, int csOn, int csOff, int csPOff,
		int csStartTN, int dataStartTN, int nDataT,
		float fracCSTMF, float fracCSPMF, float fracCtxtMF,
		float bgFreqMin, float csBGFreqMin, float ctxtFreqMin, float csTFreqMin, float csPFreqMin,
		float bgFreqMax, float csBGFreqMax, float ctxtFreqMax, float csTFreqMax, float csPFreqMax)
		:ECManagementBase(numT, iti)
{
	CRandomSFMT0 randGen(time(0));

	int numCSTMF;
	int numCSPMF;
	int numCtxtMF;

	bool *isCSTonic;
	bool *isCSPhasic;
	bool *isContext;

	csOnTime=csOn;
	csOffTime=csOff;
	csPOffTime=csPOff;

	csStartTrialN=csStartTN;
	dataStartTrialN=dataStartTN;
	numDataTrials=nDataT;

	fracCSTonicMF=fracCSTMF;
	fracCSPhasicMF=fracCSPMF;
	fracContextMF=fracCtxtMF;

	backGFreqMin=bgFreqMin;
	csBackGFreqMin=csBGFreqMin;
	contextFreqMin=ctxtFreqMin;
	csTonicFreqMin=csTFreqMin;
	csPhasicFreqMin=csPFreqMin;

	backGFreqMax=bgFreqMax;
	csBackGFreqMax=csBGFreqMax;
	contextFreqMax=ctxtFreqMax;
	csTonicFreqMax=csTFreqMax;
	csPhasicFreqMax=csPFreqMax;

	mfFreqInCSTonic=new float[numMF];
	mfFreqInCSPhasic=new float[numMF];

	isCSTonic=new bool[numMF];
	isCSPhasic=new bool[numMF];
	isContext=new bool[numMF];

	for(int i=0; i<numMF; i++)
	{
		mfFreq[i]=randGen.Random()*(backGFreqMax-backGFreqMin)+backGFreqMin;
		mfFreqInCSTonic[i]=mfFreq[i];
		mfFreqInCSPhasic[i]=mfFreq[i];

		isCSTonic[i]=false;
		isCSPhasic[i]=false;
		isContext[i]=false;
	}

	numCSTMF=fracCSTonicMF*numMF;
	numCSPMF=fracCSPhasicMF*numMF;
	numCtxtMF=fracContextMF*numMF;

	for(int i=0; i<numCSTMF; i++)
	{
		while(true)
		{
			int mfInd;

			mfInd=randGen.IRandom(0, numMF);

			if(isCSTonic[mfInd])
			{
				continue;
			}

			isCSTonic[mfInd]=true;
			break;
		}
	}

	for(int i=0; i<numCSPMF; i++)
	{
		while(true)
		{
			int mfInd;

			mfInd=randGen.IRandom(0, numMF);

			if(isCSPhasic[mfInd] || isCSTonic[mfInd])
			{
				continue;
			}

			isCSPhasic[mfInd]=true;
			break;
		}
	}

	for(int i=0; i<numCtxtMF; i++)
	{
		while(true)
		{
			int mfInd;

			mfInd=randGen.IRandom(0, numMF);

			if(isContext[mfInd] || isCSPhasic[mfInd] || isCSTonic[mfInd])
			{
				continue;
			}

			isContext[mfInd]=true;
			break;
		}
	}

	for(int i=0; i<numMF; i++)
	{
		if(isContext[i])
		{
			mfFreq[i]=randGen.Random()*(contextFreqMax-contextFreqMin)+contextFreqMin;
			mfFreqInCSTonic[i]=mfFreq[i];
			mfFreqInCSPhasic[i]=mfFreq[i];
		}

		if(isCSTonic[i])
		{
			mfFreqInCSTonic[i]=randGen.Random()*(csTonicFreqMax-csTonicFreqMin)+csTonicFreqMin;
			mfFreqInCSPhasic[i]=mfFreqInCSTonic[i];
		}

		if(isCSPhasic[i])
		{
			mfFreqInCSPhasic[i]=randGen.Random()*(csPhasicFreqMax-csPhasicFreqMin)+csPhasicFreqMin;
		}
	}

	delete[] isCSTonic;
	delete[] isCSPhasic;
	delete[] isContext;
}

ECManagementDelay::~ECManagementDelay()
{
	delete[] mfFreqInCSTonic;
	delete[] mfFreqInCSPhasic;
}

void ECManagementDelay::calcMFActivity()
{
	if(numTrials<csStartTrialN)
	{
		apMF=mf->calcActivity(mfFreq);

		return;
	}

	if(currentTime>=csOnTime && currentTime<csOffTime)
	{
		if(currentTime<csPOffTime)
		{
			apMF=mf->calcActivity(mfFreqInCSPhasic);
		}
		else
		{
			apMF=mf->calcActivity(mfFreqInCSTonic);
		}
	}
	else
	{
		apMF=mf->calcActivity(mfFreq);
	}
}

void ECManagementDelay::calcSimActivity()
{
	if(currentTime==(csOffTime-1) && currentTrial>=csStartTrialN)
	{
		simulation->updateErrDrive(0, 1);
	}
	else
	{
		simulation->updateErrDrive(0, 0);
	}
	simulation->updateMFInput(apMF);

	simulation->calcActivity();
}
