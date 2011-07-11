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

	refPFPCPopAct=new float[totalNumBins];
	curPFPCPopAct=new float[totalNumBins];

	refPFPCSynW=new float[numGR];
	curPFPCSynW=new float[numGR];

	for(int i=0; i<numGR; i++)
	{
		refPFPCSynW[i]=0.5;
		curPFPCSynW[i]=0.5;
	}

//	calcPFPCPopActivity(refPFPCPopAct, refPFPCSynW);
}

GRPSHPopAnalysis::~GRPSHPopAnalysis()
{
	delete[] refPFPCPopAct;
	delete[] curPFPCPopAct;
	delete[] refPFPCSynW;
	delete[] curPFPCSynW;
}

