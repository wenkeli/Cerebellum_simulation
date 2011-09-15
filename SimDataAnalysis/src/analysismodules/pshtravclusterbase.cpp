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

	clustersMade=false;
}

BasePSHTravCluster::~BasePSHTravCluster()
{
	for(int i=0; i<motifs.size(); i++)
	{
		delete[] motifs[i];
		delete[] motifsTotal[i];
	}
}

void BasePSHTravCluster::makeClusters()
{
	const unsigned int **data;
	float *dataRow;

	if(clustersMade)
	{
		return;
	}

	dataRow=new float[numBins];

	data=pshData->getData();
	for(int i=0; i<numCells; i++)
	{
		bool motifExists;

		if(i%(numCells/10)==0)
		{
			cout<<"making clusters: "<<((float)i)/numCells*100<<" % done"<<endl;
		}
		for(int j=0; j<numBins; j++)
		{
			dataRow[j]=data[j][i];
		}

		motifExists=false;
		for(int j=0; j<motifs.size(); j++)
		{
			if(!isDifferent(dataRow, motifs[j]))
			{
				insertInMotif(dataRow, j, i);
				motifExists=true;
				break;
			}
		}
		if(!motifExists)
		{
			addMotif(dataRow, i);
		}
	}

	clustersMade=true;

	cout<<"num clusters: "<<getNumClusters()<<endl;

	delete[] dataRow;
}

bool BasePSHTravCluster::isAnalyzed()
{
	return clustersMade;
}

unsigned int BasePSHTravCluster::getNumClusters()
{
	return motifs.size();
}

unsigned int BasePSHTravCluster::getNumClusterCells(unsigned int clusterN)
{
	if(clusterN>=motifs.size())
	{
		clusterN=motifs.size()-1;
	}
	return clusterIndices[clusterN].size();
}

QPixmap *BasePSHTravCluster::viewCluster(unsigned int clusterN)
{

	if(!clustersMade)
	{
		return NULL;
	}

	if(clusterN>=motifs.size())
	{
		clusterN=motifs.size()-1;
	}

	return pshData->paintPSH(motifs[clusterN]);
}

QPixmap *BasePSHTravCluster::viewClusterCell(unsigned int clusterN, unsigned int clusterCellN)
{

	if(!clustersMade)
	{
		return NULL;
	}

	if(clusterN>=motifs.size())
	{
		clusterN=motifs.size()-1;
	}

	if(clusterCellN>=clusterIndices[clusterN].size())
	{
		clusterCellN=clusterIndices[clusterN].size()-1;
	}

	return pshData->paintPSHInd(clusterIndices[clusterN][clusterCellN]);
}

void BasePSHTravCluster::addMotif(float *row, int cellInd)
{
	float *dataRow;
	unsigned long *dataRowTotal;
	vector<unsigned int> inds;

	dataRow=new float[numBins];
	dataRowTotal= new unsigned long[numBins];

	for(int i=0; i<numBins; i++)
	{
		dataRow[i]=row[i];
		dataRowTotal[i]=row[i];
	}

	motifs.push_back(dataRow);
	motifsTotal.push_back(dataRowTotal);

	inds.push_back(cellInd);
	clusterIndices.push_back(inds);
}

void BasePSHTravCluster::insertInMotif(float *row, int motifInd, int cellInd)
{
	int numCells;

	clusterIndices[motifInd].push_back(cellInd);
	numCells=clusterIndices[motifInd].size();

	for(int i=0; i<numBins; i++)
	{
		motifsTotal[motifInd][i]+=row[i];
		motifs[motifInd][i]=((float)motifsTotal[motifInd][i])/numCells;
	}
}

