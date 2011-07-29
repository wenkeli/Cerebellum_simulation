/*
 * spikerateanalysis.cpp
 *
 *  Created on: Jul 29, 2011
 *      Author: consciousness
 */

#include "../../includes/analysismodules/spikerateanalysis.h"

SpikeRateAnalysis::SpikeRateAnalysis(PSHData *pshData)
{
	psh=pshData->getData();
	numTrials=pshData->getNumTrials();
	preStimNumBins=pshData->getPreStimNumBins();
	stimNumBins=pshData->getStimNumBins();
	binTimeSize=pshData->getBinTimeSize();
	numCells=pshData->getCellNum();

	preStimSR=new float[numCells];
	stimulusSR=new float[numCells];
	postStimSR=new float[numCells];
}

SpikeRateAnalysis::~SpikeRateAnalysis()
{
	delete[] preStimSR;
	delete[] stimulusSR;
	delete[] postStimSR;
}

void SpikeRateAnalysis::calcSpikeRates()
{
	calcSR(preStimSR, 0, preStimNumBins);
	calcSR(stimulusSR, preStimNumBins, stimNumBins);
	calcSR(postStimSR, preStimNumBins+stimNumBins, postStimNumBins);
}

void SpikeRateAnalysis::exportSpikeRates(ofstream &outfile)
{
	for(int i=0; i<numCells; i++)
	{
		outfile<<preStimSR[i]<<", "<<stimulusSR[i]<<", "<<postStimSR[i]<<endl;
	}
}

void SpikeRateAnalysis::calcSR(float *spikeRates, unsigned int startBinN, unsigned nBins)
{
	for(int i=0; i<numCells; i++)
	{
		spikeRates[i]=0;
	}

	for(int i=startBinN; i<startBinN+nBins; i++)
	{
		for(int j=0; j<numCells; j++)
		{
			spikeRates[j]+=psh[i][j];
		}

	}

	for(int i=0; i<numCells; i++)
	{
		spikeRates[i]=spikeRates[i]/(numTrials*(nBins*binTimeSize)/1000.0f);
	}
}
