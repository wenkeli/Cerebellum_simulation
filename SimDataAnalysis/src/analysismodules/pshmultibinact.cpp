/*
 * pshmultibinact.cpp
 *
 *  Created on: May 9, 2012
 *      Author: consciousness
 */

#include "../../includes/analysismodules/pshmultibinact.h"

using namespace std;
PSHMultiBinAct::PSHMultiBinAct(unsigned int startBN, unsigned int endBN, PSHData *psh)
{
	int maxPSHBinN;

	maxPSHBinN=psh->getTotalNumBins();

	if(startBN>=maxPSHBinN || endBN>=maxPSHBinN || startBN>endBN)
	{
		cerr<<"PSHMultiBinN: bad bin numbers "<<maxPSHBinN<<" "<<startBN<<" "<<endBN<<endl;
		delete this;
		return;
	}

	startBinN=startBN;
	endBinN=endBN;

	numTrials=psh->getNumTrials();
	numCells=psh->getCellNum();
	numBins=endBinN-startBinN+1;
	totalTimeLen=numBins*psh->getBinTimeSize();

	actData=new float[numCells];
	actDataNormToTrials=new float[numCells];

	for(int i=0; i<numCells; i++)
	{
		actData[i]=0;
	}

	for(int i=startBN; i<=endBN; i++)
	{
		const unsigned int *pshBin;

		pshBin=psh->getDataRow(i);
		for(int j=0; j<numCells; j++)
		{
			actData[j]=actData[j]+pshBin[j];
		}
	}

	for(int i=0; i<numCells; i++)
	{
		actDataNormToTrials[i]=actData[i]/numTrials;
	}
}

PSHMultiBinAct::~PSHMultiBinAct()
{
	delete actData;
	delete actDataNormToTrials;
}

int PSHMultiBinAct::getNumTrials()
{
	return numTrials;
}

int PSHMultiBinAct::getNumCells()
{
	return numCells;
}

int PSHMultiBinAct::getNumBins()
{
	return numBins;
}

int PSHMultiBinAct::getStartBinN()
{
	return startBinN;
}

int PSHMultiBinAct::getEndBinN()
{
	return endBinN;
}

int PSHMultiBinAct::getTotalTimeLen()
{
	return totalTimeLen;
}

const float* PSHMultiBinAct::getActData()
{
	return (const float*)actData;
}

const float* PSHMultiBinAct::getActDataNormToTrials()
{
	return (const float*)actDataNormToTrials;
}
