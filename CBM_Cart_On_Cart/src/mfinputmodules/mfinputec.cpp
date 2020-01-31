/*
 * mfinputec.cpp
 *
 *  Created on: Apr 29, 2011
 *      Author: consciousness
 */

#include "../../includes/mfinputmodules/mfinputec.h"
#include "../../includes/globalvars.h"

ECMFInput::ECMFInput(unsigned int nmf, unsigned int csSTSN, unsigned int csETSN, unsigned int csPhaETSN,
		float csTonicP, float csPhasicP, float contextP, float ts, float tsus):BaseMFInput(nmf, ts, tsus)
{
	csTonicMFProportion=csTonicP;
	csPhasicMFProportion=csPhasicP;
	contextMFProportion=contextP;

	csStart=csSTSN*timeStepSize*tsUnitInS;
	csEnd=csETSN*timeStepSize*tsUnitInS;
	csPhasicMFEnd=csPhaETSN*timeStepSize*tsUnitInS;

	threshDecay=1-exp(-(ts/threshDecayT));

	mfType=new char[nmf];
	bgFreq=new float[nmf];
	incFreq=new float[nmf];
	incStart=new float[nmf];
	incEnd=new float[nmf];

	threshold=new float[nmf];

	memset((void *)mfType, mfTypeDefNonAct, nmf*sizeof(char));

	assignMF(mfTypeDefContext, nmf*contextMFProportion);
	assignMF(mfTypeDefCSTonic, nmf*csTonicMFProportion);
	assignMF(mfTypedefCSPhasic, nmf*csPhasicMFProportion);

	for(int i=0; i<nmf; i++)
	{
		bgFreq[i]=randGen->Random()*(bgFreqMax-bgFreqMin)+bgFreqMin;
		incFreq[i]=0;
		incStart[i]=0;
		incEnd[i]=0;

		threshold[i]=1;
		if(mfType[i]==mfTypeDefContext)
		{
			bgFreq[i]=randGen->Random()*(bgContextFreqMax-bgContextFreqMin)+bgContextFreqMin;
		}
		else if(mfType[i]==mfTypeDefCSTonic)
		{
			bgFreq[i]=randGen->Random()*(bgCSFreqMax-bgCSFreqMin)+bgCSFreqMin;
			incFreq[i]=csTonicIncFreq;
			incStart[i]=csStart;
			incEnd[i]=csEnd;
		}
		else if(mfType[i]==mfTypedefCSPhasic)
		{
			bgFreq[i]=randGen->Random()*(bgCSFreqMax-bgCSFreqMin)+bgCSFreqMin;
			incFreq[i]=csPhasicIncFreq;
			incStart[i]=csStart;
			incEnd[i]=csPhasicMFEnd;
		}

		bgFreq[i]=bgFreq[i]*(timeStepSize*tsUnitInS);
		incFreq[i]=incFreq[i]*(timeStepSize*tsUnitInS);
	}
}

ECMFInput::ECMFInput(ifstream &infile):BaseMFInput(infile)
{
	infile.read((char *)&threshDecay, sizeof(float));
	infile.read((char *)&contextMFProportion, sizeof(float));
	infile.read((char *)&csTonicMFProportion, sizeof(float));
	infile.read((char *)&csPhasicMFProportion, sizeof(float));
	infile.read((char *)&csStart, sizeof(float));
	infile.read((char *)&csEnd, sizeof(float));
	infile.read((char *)&csPhasicMFEnd, sizeof(float));

	mfType=new char[numMF];
	bgFreq=new float[numMF];
	incFreq=new float[numMF];
	incStart=new float[numMF];
	incEnd=new float[numMF];
	threshold=new float[numMF];

	infile.read((char *)mfType, numMF*sizeof(char));
	infile.read((char *)bgFreq, numMF*sizeof(float));
	infile.read((char *)incFreq, numMF*sizeof(float));
	infile.read((char *)incStart, numMF*sizeof(float));
	infile.read((char *)incEnd, numMF*sizeof(float));
	infile.read((char *)threshold, numMF*sizeof(float));

	cout<<"mf mod read: "<<numMF<<" "<<timeStepSize<<" "<<tsUnitInS<<" "<<threshDecay<<" "
			<<contextMFProportion<<" "<<csTonicMFProportion<<" "<<csPhasicMFProportion<<" "
			<<csStart<<" "<<csEnd<<" "<<csPhasicMFEnd<<" "<<(int)mfType[0]<<" "
			<<bgFreq[0]<<" "<<incFreq[0]<<" "<<incStart[0]<<" "<<incEnd[0]<<" "<<threshold[0]<<endl;
}

void ECMFInput::exportState(ofstream &outfile)
{
	BaseMFInput::exportState(outfile);

	outfile.write((char *)&threshDecay, sizeof(float));
	outfile.write((char *)&contextMFProportion, sizeof(float));
	outfile.write((char *)&csTonicMFProportion, sizeof(float));
	outfile.write((char *)&csPhasicMFProportion, sizeof(float));
	outfile.write((char *)&csStart, sizeof(float));
	outfile.write((char *)&csEnd, sizeof(float));
	outfile.write((char *)&csPhasicMFEnd, sizeof(float));

	outfile.write((char *)mfType, numMF*sizeof(char));
	outfile.write((char *)bgFreq, numMF*sizeof(float));
	outfile.write((char *)incFreq, numMF*sizeof(float));
	outfile.write((char *)incStart, numMF*sizeof(float));
	outfile.write((char *)incEnd, numMF*sizeof(float));
	outfile.write((char *)threshold, numMF*sizeof(float));
}

ECMFInput::~ECMFInput()
{
	delete[] mfType;
	delete[] bgFreq;
	delete[] incFreq;
	delete[] incStart;
	delete[] incEnd;
	delete[] threshold;
}

void ECMFInput::assignMF(char assignType, unsigned int assignNum)
{
	int mfAssignedCount;

	mfAssignedCount=0;
	while(mfAssignedCount<assignNum)
	{
		int mfInd;

		mfInd=randGen->IRandom(0, numMF);
		if(mfType[mfInd]==mfTypeDefNonAct)
		{
			mfType[mfInd]=assignType;
			mfAssignedCount++;
		}
	}
}

void ECMFInput::calcActivity(unsigned int tsN, unsigned int trial)
{
	float timeInS=tsN*timeStepSize*tsUnitInS;
	for(int i=0; i<numMF; i++)
	{
		bool incOn;
		threshold[i]=threshold[i]+(1-threshold[i])*threshDecay;
		incOn=(timeInS>=incStart[i] && timeInS<incEnd[i]);

		apMF[i]=randGen->Random()<(incOn*incFreq[i]+bgFreq[i])*threshold[i];
		threshold[i]=(!apMF[i])*threshold[i];
	}
}
