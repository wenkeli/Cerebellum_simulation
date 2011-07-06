/*
 * psh.cpp
 *
 *  Created on: Jul 6, 2011
 *      Author: consciousness
 */

#include "../../includes/datamodules/psh.h"

PSHAnalysis::PSHAnalysis(ifstream &infile)
{
	infile.read((char *)&numCells, sizeof(unsigned int));
	infile.read((char *)&numBins, sizeof(unsigned int));
	infile.read((char *)&binTimeSize, sizeof(unsigned int));
	infile.read((char *)&apBufTimeSize, sizeof(unsigned int));
	infile.read((char *)&numBinsInBuf, sizeof(unsigned int));
	infile.read((char *)&numTrials, sizeof(unsigned int));

	pshData=new unsigned int *[numBins];
	pshData[0]=new unsigned int[numBins*numCells];
	for(int i=1; i<numBins; i++)
	{
		pshData[i]=&(pshData[0][numCells*i]);
	}

	infile.read((char *)pshData[0], numBins*numCells*sizeof(unsigned int));

	currBinN=0;

	pshBinMaxVal=0;

	for(int i=0; i<numBins; i++)
	{
		for(int j=0; j<numCells; j++)
		{
			if(pshData[i][j]>pshBinMaxVal)
			{
				pshBinMaxVal=pshData[i][j];
			}
		}
	}
}


PSHAnalysis::~PSHAnalysis()
{
	delete[] pshData[0];
	delete[] pshData;
}

void PSHAnalysis::exportPSH(ofstream &outfile)
{
	outfile.write((char *)&numCells, sizeof(unsigned int));
	outfile.write((char *)&numBins, sizeof(unsigned int));
	outfile.write((char *)&binTimeSize, sizeof(unsigned int));
	outfile.write((char *)&apBufTimeSize, sizeof(unsigned int));
	outfile.write((char *)&numBinsInBuf, sizeof(unsigned int));
	outfile.write((char *)&numTrials, sizeof(unsigned int));
	outfile.write((char *)pshData[0], numCells*numBins*sizeof(unsigned int));
}

void PSHAnalysis::paintPSHPop(QPixmap *paintBuf)
{

}

void PSHAnalysis::paintPSHInd(QPixmap *paintBuf)
{

}


