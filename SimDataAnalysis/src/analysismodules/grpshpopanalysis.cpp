/*
 * grpopanalysis.cpp
 *
 *  Created on: Jul 7, 2011
 *      Author: consciousness
 */

#include "../../includes/analysismodules/grpshpopanalysis.h"

GRPSHPopAnalysis::GRPSHPopAnalysis(PSHData *grData)
{
	grPSH=grData->getData();
	numTrials=grData->getNumTrials();
	preStimNumBins=grData->getPreStimNumBins();
	stimNumBins=grData->getStimNumBins();
	postStimNumBins=grData->getPostStimNumBins();
	totalNumBins=grData->getTotalNumBins();
	binTimeSize=grData->getBinTimeSize();
	pshMaxVal=grData->getPSHBinMaxVal();
	numGR=grData->getCellNum();

	grPSHNormalized=new float *[totalNumBins];
	grPSHNormalized[0]=new float[totalNumBins*numGR];
	for(int i=1; i<totalNumBins; i++)
	{
		grPSHNormalized[i]=&(grPSHNormalized[0][i*numGR]);
	}

	refPFPCPopAct=new float[totalNumBins];
	curPFPCPopAct=new float[totalNumBins];

	refPFPCSynW=new float[numGR];
	curPFPCSynW=new float[numGR];

	for(int i=0; i<numGR; i++)
	{
		refPFPCSynW[i]=0.5;
		curPFPCSynW[i]=0.5;
	}

	for(int i=0; i<totalNumBins; i++)
	{
		for(int j=0; j<numGR; j++)
		{
			grPSHNormalized[i][j]=((float)grPSH[i][j])/(numTrials*binTimeSize);
		}
	}

	calcPFPCPopActivity(refPFPCPopAct, refPFPCSynW);
}

GRPSHPopAnalysis::~GRPSHPopAnalysis()
{
	delete[] grPSHNormalized[0];
	delete[] grPSHNormalized;

	delete[] refPFPCPopAct;
	delete[] curPFPCPopAct;
	delete[] refPFPCSynW;
	delete[] curPFPCSynW;
}

void GRPSHPopAnalysis::calcPFPCPopActivity(float *activity, float *pfPCSynW)
{
	for(int i=0; i<totalNumBins; i++)
	{
		float tempSum;

		tempSum=0;
		for(int j=0; j<numGR; j++)
		{
			tempSum+=grPSHNormalized[i][j]*pfPCSynW[j];
		}
		activity[i]=tempSum;
	}
}

void GRPSHPopAnalysis::calcPFPCPlast(unsigned int usTime)
{
	if(usTime>stimNumBins*binTimeSize)
	{
		cerr<<"invalid us time, must be between 0 and "<<stimNumBins*binTimeSize<<endl;
		return;
	}

	for(int i=0; i<numGR; i++)
	{
		curPFPCSynW[i]=0.5;
	}

	for(int i=0; i<100; i++)
	{
		runPFPCPlastIteration(usTime);
		cout<<"PFPC plast iteration "<<i<<endl;
	}
}

void GRPSHPopAnalysis::runPFPCPlastIteration(unsigned int usTime)
{
	unsigned int usBinN;

	int usLTDSBinN;
	int usLTDEBinN;

	usBinN=usTime/binTimeSize+preStimNumBins;
	usLTDSBinN=(usTime-200)/binTimeSize+preStimNumBins;
	usLTDEBinN=(usTime-100)/binTimeSize+preStimNumBins;

	calcPFPCPopActivity(curPFPCPopAct, curPFPCSynW);

	for(int i=0; i<totalNumBins; i++)
	{
		if(i>=usLTDSBinN && i<usLTDEBinN)
		{
			doPFPCPlast(-0.0001*binTimeSize, grPSHNormalized[i], curPFPCSynW);
		}
		else
		{
			float ltpStep;

			ltpStep=(refPFPCPopAct[i]-curPFPCPopAct[i])/refPFPCPopAct[i];
			ltpStep=0.0001*binTimeSize*(ltpStep>0)*ltpStep;
			doPFPCPlast(ltpStep, grPSHNormalized[i], curPFPCSynW);
		}
	}
}

void GRPSHPopAnalysis::doPFPCPlast(float plastStep, const float *pshRow, float *pfPCSynW)
{
	for(int i=0; i<numGR; i++)
	{
		pfPCSynW[i]+=plastStep*pshRow[i];
		pfPCSynW[i]=(pfPCSynW[i]>0)*pfPCSynW[i];
		pfPCSynW[i]=(pfPCSynW[i]>=1)+(pfPCSynW[i]<1)*pfPCSynW[i];
	}
}

void GRPSHPopAnalysis::exportPFPCPlastAct(ofstream &outfile)
{
	for(int i=0; i<totalNumBins; i++)
	{
		outfile<<i*binTimeSize<<", "<<refPFPCPopAct[i]<<", "<<curPFPCPopAct[i]<<endl;
	}
}
