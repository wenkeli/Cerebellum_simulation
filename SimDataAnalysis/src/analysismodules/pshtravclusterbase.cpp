/*
 * pshtravclusterbase.cpp
 *
 *  Created on: Sep 13, 2011
 *      Author: consciousness
 */

#include "../../includes/analysismodules/pshtravclusterbase.h"

BasePSHTravCluster::BasePSHTravCluster(PSHData *data)
{
	pshData=data;
	numBins=pshData->getTotalNumBins();
	numCells=pshData->getCellNum();
}

void BasePSHTravCluster::makeClusters()
{
	const unsigned int **data;
	float *dataRow;

	dataRow=new float[numBins];

	data=pshData->getData();
	for(int i=0; i<numCells; i++)
	{
		bool motifExists;

		for(int j=0; j<numBins; j++)
		{
			dataRow[j]=data[j][i];
		}

		motifExists=false;
		for(int j=0; j<motifs.size(); j++)
		{
			if(isSimilar(dataRow, motifs[j]))
			{
				insertInMotif(j, i);
				motifExists=true;
				break;
			}
		}
		if(!motifExists)
		{
			addMotif(dataRow, i);
		}
	}

	delete[] dataRow;
}

BasePSHTravCluster::~BasePSHTravCluster()
{
	for(int i=0; i<motifs.size(); i++)
	{
		delete[] motifs[i];
	}
}

void BasePSHTravCluster::addMotif(float *row, int cellInd)
{
	float *dataRow;
	vector<unsigned int> inds;

	dataRow=new float[numBins];

	for(int i=0; i<numBins; i++)
	{
		dataRow[i]=row[i];
	}

	motifs.push_back(dataRow);

	inds.push_back(cellInd);
	clusterIndices.push_back(inds);
}

void BasePSHTravCluster::insertInMotif(int motifInd, int cellInd)
{
	clusterIndices[motifInd].push_back(cellInd);
}

