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
}
