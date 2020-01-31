/*
 * ecmfpopulation.cpp
 *
 *  Created on: Jul 13, 2014
 *      Author: consciousness
 */

#include "../CBMToolsInclude/ecmfpopulation.h"
using namespace std;

ECMFPopulation::ECMFPopulation(ECTrialsData *data)
{
	PeriStimHistFloat *mfPSH;

	int msPerBin;
	int numTrials;
	float scaling;
	int binsPreTrial;
	int binsTrial;



	mfPSH=data->getPSH("mf");

	numMF=mfPSH->getNumCells();
	msPerBin=mfPSH->getBinWidthInMS();
	numTrials=mfPSH->getNumTrials();
	scaling=numTrials*(msPerBin/1000.0);
	binsPreTrial=data->getMSPreTrial()/msPerBin;
	binsTrial=data->getMSTrial()/msPerBin;


	mfFreqBG=new float[numMF];
	mfFreqInCSPhasic=new float[numMF];
	mfFreqInCSTonic=new float[numMF];

	for(int i=0; i<numMF; i++)
	{
		vector<float> cellPSH;
		cellPSH=mfPSH->getCellPSH(i);

		float binSum;

		binSum=0;
		for(int j=2; j<binsPreTrial-2; j++)
		{
			binSum+=cellPSH[j];
		}
		mfFreqBG[i]=binSum/((binsPreTrial-4)*scaling);

		binSum=0;
		for(int j=binsPreTrial+1; j<binsPreTrial+3; j++)
		{
			binSum+=cellPSH[j];
		}
		mfFreqInCSPhasic[i]=binSum/(2*scaling);

		binSum=0;
		for(int j=binsPreTrial+binsTrial-6; j<binsPreTrial+binsTrial-1; j++)
		{
			binSum+=cellPSH[j];
		}
		mfFreqInCSTonic[i]=binSum/(5*scaling);
	}
}

ECMFPopulation::ECMFPopulation(int numMF, int randSeed, float fracCSTMF, float fracCSPMF, float fracCtxtMF,
		float bgFreqMin, float csBGFreqMin, float ctxtFreqMin, float csTFreqMin, float csPFreqMin,
		float bgFreqMax, float csBGFreqMax, float ctxtFreqMax, float csTFreqMax, float csPFreqMax)
{
	CRandomSFMT0 *randGen;

	bool *isCSTonic;
	bool *isCSPhasic;
	bool *isContext;

	int numCSTMF;
	int numCSPMF;
	int numCtxtMF;

	randGen=new CRandomSFMT0(randSeed);
	this->numMF=numMF;

	mfFreqBG=new float[numMF];
	mfFreqInCSPhasic=new float[numMF];
	mfFreqInCSTonic=new float[numMF];

	isCSTonic=new bool[numMF];
	isCSPhasic=new bool[numMF];
	isContext=new bool[numMF];

	for(int i=0; i<numMF; i++)
	{
		mfFreqBG[i]=randGen->Random()*(bgFreqMax-bgFreqMin)+bgFreqMin;
		mfFreqInCSTonic[i]=mfFreqBG[i];
		mfFreqInCSPhasic[i]=mfFreqBG[i];

		isCSTonic[i]=false;
		isCSPhasic[i]=false;
		isContext[i]=false;
	}

	numCSTMF=fracCSTMF*numMF;
	numCSPMF=fracCSPMF*numMF;
	numCtxtMF=fracCtxtMF*numMF;

	for(int i=0; i<numCSTMF; i++)
	{
		while(true)
		{
			int mfInd;

			mfInd=randGen->IRandom(0, numMF-1);

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

			mfInd=randGen->IRandom(0, numMF-1);

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

			mfInd=randGen->IRandom(0, numMF-1);

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
			mfFreqBG[i]=randGen->Random()*(ctxtFreqMax-ctxtFreqMin)+ctxtFreqMin;
			mfFreqInCSTonic[i]=mfFreqBG[i];
			mfFreqInCSPhasic[i]=mfFreqBG[i];
		}

		if(isCSTonic[i])
		{
			mfFreqBG[i]=randGen->Random()*(csBGFreqMax-csBGFreqMin)+csBGFreqMin;
			mfFreqInCSTonic[i]=randGen->Random()*(csTFreqMax-csTFreqMin)+csTFreqMin;
			mfFreqInCSPhasic[i]=mfFreqInCSTonic[i];
		}

		if(isCSPhasic[i])
		{
			mfFreqBG[i]=randGen->Random()*(csBGFreqMax-csBGFreqMin)+csBGFreqMin;
			mfFreqInCSTonic[i]=mfFreqBG[i];
			mfFreqInCSPhasic[i]=randGen->Random()*(csPFreqMax-csPFreqMin)+csPFreqMin;
		}
	}

	delete[] isCSTonic;
	delete[] isCSPhasic;
	delete[] isContext;

	delete randGen;
}

ECMFPopulation::ECMFPopulation(fstream &infile)
{
	infile.read((char *)&numMF, sizeof(numMF));
//	infile>>numMF;

	mfFreqBG=new float[numMF];
	mfFreqInCSTonic=new float[numMF];
	mfFreqInCSPhasic=new float[numMF];

//	for(int i=0; i<numMF; i++)
//	{
//		infile>>mfFreqBG[i];
//		infile>>mfFreqInCSTonic[i];
//		infile>>mfFreqInCSPhasic[i];
//	}

	infile.read((char *)mfFreqBG, numMF*sizeof(float));
	infile.read((char *)mfFreqInCSTonic, numMF*sizeof(float));
	infile.read((char *)mfFreqInCSPhasic, numMF*sizeof(float));
}

ECMFPopulation::~ECMFPopulation()
{
	delete[] mfFreqBG;
	delete[] mfFreqInCSTonic;
	delete[] mfFreqInCSPhasic;
}

void ECMFPopulation::writeToFile(fstream &outfile)
{
//	outfile<<numMF<<endl;
//
//	for(int i=0; i<numMF; i++)
//	{
//		outfile<<mfFreqBG[i]<<" ";
//		outfile<<mfFreqInCSTonic[i]<<" ";
//		outfile<<mfFreqInCSPhasic[i]<<endl;
//	}
	outfile.write((char *)&numMF, sizeof(numMF));

	outfile.write((char *)mfFreqBG, numMF*sizeof(float));
	outfile.write((char *)mfFreqInCSTonic, numMF*sizeof(float));
	outfile.write((char *)mfFreqInCSPhasic, numMF*sizeof(float));
}

float *ECMFPopulation::getMFBG()
{
	return mfFreqBG;
}
float *ECMFPopulation::getMFInCSTonic()
{
	return mfFreqInCSTonic;
}
float *ECMFPopulation::getMFFreqInCSPhasic()
{
	return mfFreqInCSPhasic;
}
