/*
 * psh.cpp
 *
 *  Created on: Jun 30, 2011
 *      Author: consciousness
 */


#include "../../includes/analysismodules/psh.h"

PSHAnalysis::PSHAnalysis(unsigned int nCells, const unsigned int *buf,
		unsigned int preSNBins, unsigned int sNBins, unsigned int postSNBins,
		unsigned int binSize, unsigned int bufSize, unsigned int nBinsInBuf)
{
	numCells=nCells;
	preStimNumBins=preSNBins;
	stimNumBins=sNBins;
	postStimNumBins=postSNBins;
	totalNumBins=preStimNumBins+stimNumBins+postStimNumBins;
	binTimeSize=binSize;
	apBufTimeSize=bufSize;
	numBinsInBuf=nBinsInBuf;
	numTrials=0;

	apBuf=buf;

	pshData=new unsigned int *[totalNumBins];
	pshData[0]=new unsigned int[totalNumBins*numCells];
	for(int i=1; i<totalNumBins; i++)
	{
		pshData[i]=&(pshData[0][numCells*i]);
	}

	for(int i=0; i<totalNumBins*numCells; i++)
	{
		pshData[0][i]=0;
	}

	currBinN=0;
}

PSHAnalysis::PSHAnalysis(ifstream &infile, const unsigned int *buf)
{
	apBuf=buf;

	infile.read((char *)&numCells, sizeof(unsigned int));
	infile.read((char *)&preStimNumBins, sizeof(unsigned int));
	infile.read((char *)&stimNumBins, sizeof(unsigned int));
	infile.read((char *)&postStimNumBins, sizeof(unsigned int));
	infile.read((char *)&totalNumBins, sizeof(unsigned int));
	infile.read((char *)&binTimeSize, sizeof(unsigned int));
	infile.read((char *)&apBufTimeSize, sizeof(unsigned int));
	infile.read((char *)&numBinsInBuf, sizeof(unsigned int));
	infile.read((char *)&numTrials, sizeof(unsigned int));

	pshData=new unsigned int *[totalNumBins];
	pshData[0]=new unsigned int[totalNumBins*numCells];
	for(int i=1; i<totalNumBins; i++)
	{
		pshData[i]=&(pshData[0][numCells*i]);
	}

	infile.read((char *)pshData[0], totalNumBins*numCells*sizeof(unsigned int));

	currBinN=0;
}


PSHAnalysis::~PSHAnalysis()
{
	delete[] pshData[0];
	delete[] pshData;
}

void PSHAnalysis::updatePSH()
{
	int extrashift;

	extrashift=apBufTimeSize-(numBinsInBuf*binTimeSize);

	for(int i=0; i<numBinsInBuf; i++)
	{
		for(int j=0; j<numCells; j++)
		{
			unsigned int tempCount;
			unsigned int tempBuf;

			tempCount=0;
			tempBuf=apBuf[j]<<(binTimeSize*i+extrashift);
			for(int k=0; k<binTimeSize; k++)
			{
				tempCount+=(tempBuf&0x80000000)>0;
				tempBuf=tempBuf<<1;
			}
			pshData[i+currBinN][j]+=tempCount;
		}
	}

	currBinN=currBinN+numBinsInBuf;

	if(currBinN>=totalNumBins)
	{
		numTrials++;
		currBinN=0;
	}
//	currBinN=currBinN%numBins;
}

void PSHAnalysis::resetCurrentBinN()
{
	currBinN=0;
}

void PSHAnalysis::exportPSH(ofstream &outfile)
{
	outfile.write((char *)&numCells, sizeof(unsigned int));
	outfile.write((char *)&preStimNumBins, sizeof(unsigned int));
	outfile.write((char *)&stimNumBins, sizeof(unsigned int));
	outfile.write((char *)&postStimNumBins, sizeof(unsigned int));
	outfile.write((char *)&totalNumBins, sizeof(unsigned int));
	outfile.write((char *)&binTimeSize, sizeof(unsigned int));
	outfile.write((char *)&apBufTimeSize, sizeof(unsigned int));
	outfile.write((char *)&numBinsInBuf, sizeof(unsigned int));
	outfile.write((char *)&numTrials, sizeof(unsigned int));
	outfile.write((char *)pshData[0], numCells*totalNumBins*sizeof(unsigned int));
}
