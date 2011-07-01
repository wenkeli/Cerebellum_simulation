/*
 * psh.cpp
 *
 *  Created on: Jul 1, 2011
 *      Author: consciousness
 */

#include "../includes/psh.h"

template<typename Type>
PSHData<Type>::PSHData(ifstream &infile)
{
	unsigned int dummy;

	infile.read((char *)&numCells, sizeof(unsigned int));
	infile.read((char *)&numBins, sizeof(unsigned int));
	infile.read((char *)&binTimeSize, sizeof(unsigned int));
	infile.read((char *)&dummy, sizeof(unsigned int));
	infile.read((char *)&dummy, sizeof(unsigned int));
	infile.read((char *)&numTrials, sizeof(unsigned int));

	pshData=new Type *[numCells];
	pshData[0]=new Type[numCells*numBins];
	for(int i=1; i<numCells; i++)
	{
		pshData[i]=&(pshData[0][numBins*i]);
	}

	infile.read((char *)pshData[0], numCells*numBins*sizeof(Type));

	maxBinVal=0;
	for(int i=0; i<numCells; i++)
	{
		for(int j=0; j<numBins; j++)
		{
			if(pshData[i][j]>maxBinVal)
			{
				maxBinVal=pshData[i][j];
			}
		}
	}
}
