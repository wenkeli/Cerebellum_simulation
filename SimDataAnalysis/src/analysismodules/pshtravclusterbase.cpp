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

	unsigned int numClusters;

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

	numClusters=getNumClusters();
	while(true)
	{
		unsigned int mergedNumClusters;

		mergeMotifs();

		mergedNumClusters=getNumClusters();

		if(mergedNumClusters==numClusters)
		{
			break;
		}
		numClusters=mergedNumClusters;
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
	return motifCellIndices[clusterN].size();
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

	if(clusterCellN>=motifCellIndices[clusterN].size())
	{
		clusterCellN=motifCellIndices[clusterN].size()-1;
	}

	return pshData->paintPSHInd(motifCellIndices[clusterN][clusterCellN]);
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
	motifCellIndices.push_back(inds);
}

void BasePSHTravCluster::insertInMotif(float *row, int motifInd, int cellInd)
{
	int numCells;

	motifCellIndices[motifInd].push_back(cellInd);
	numCells=motifCellIndices[motifInd].size();

	for(int i=0; i<numBins; i++)
	{
		motifsTotal[motifInd][i]+=row[i];
		motifs[motifInd][i]=((float)motifsTotal[motifInd][i])/numCells;
	}
}

void BasePSHTravCluster::mergeMotifs()
{
	vector<float *> mergedMotifs;
	vector<unsigned long *> mergedMotifsTotal;
	vector< vector<unsigned int> > mergedMotifIndices;

	for(int i=0; i<getNumClusters(); i++)
	{
		bool toMerge;

		toMerge=false;
		for(int j=0; j<mergedMotifs.size(); j++)
		{
			if(!isDifferentMotif())
			{
				toMerge=true;
				doMotifsMerge();
				break;
			}
		}

		if(!toMerge)
		{
			insertMergeMotif();
		}
	}


}



