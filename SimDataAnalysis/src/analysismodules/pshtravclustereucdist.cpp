/*
 * pshtravclustereucdist.cpp
 *
 *  Created on: Nov 29, 2011
 *      Author: consciousness
 */

#include "../../includes/analysismodules/pshtravclustereucdist.h"

EucDistPSHTravCluster::EucDistPSHTravCluster(PSHData *data, float thresh, unsigned int distNumBins)
	:BasePSHTravCluster(data)
{
	threshP=thresh;
	pshNumBins=distNumBins;

	eculideanPDF=new float[pshNumBins];
	eculideanCDF=new float[pshNumBins];

	generateDist();
}

EucDistPSHTravCluster::~EucDistPSHTravCluster()
{
	delete[] eculideanPDF;
	delete[] eculideanCDF;
	delete[] distances;
}

bool EucDistPSHTravCluster::isDifferent(float *psh1, float *psh2)
{
	float distance;

	distance=calcEuclideanDist(psh1, psh2);

	if(dist>threshVal)
	{
		return true;
	}

	return false;
}

void EucDistPSHTravCluster::generateDist()
{
	const unsigned int **data;
	float *pshRow1;
	float *pshRow2;
	float maxDistance;
	float minDistance;
	CRandomSFMT0 randGen((int)time(NULL));

	//	randGen=new CRandomSFMT0((int)time(NULL));
	data=pshData->getData();

	distances=new float[numCells];
	pshRow1=new float[pshNumBins];
	pshRow2=new float[pshNumBins];

	for(int i=0; i<numCells; i++)
	{
		int cellInd1, cellInd2;
		cellInd1=randGen.IRandom(0, numCells-1);
		cellInd2=randGen.IRandom(0, numCells-1);

		while(cellInd2==cellInd1)
		{
			cellInd2=randGen.IRandom(0, numCells-1);
		}

		for(int j=0; j<pshNumBins; j++)
		{
			pshRow1[j]=data[j][cellInd1];
			pshRow2[j]=data[j][cellInd2];
		}
		distances[i]=calcEuclideanDist(pshRow1, pshRow2);
	}

	maxDistance=distances[0];
	minDistance=distances[0];
	for(int i=1; i<numCells; i++)
	{
		if(distances[i]>maxDistance)
		{
			maxDistance=distances[i];
		}
		if(distances[i]<minDistance)
		{
			minDistance=distances[i];
		}
	}


}

float EucDistPSHTravCluster::calcEuclideanDist(float *psh1, float *psh2)
{
	float distance;

	distance=0;

	for(int i=0; i<pshNumBins; i++)
	{
		distance+=(psh1[i]-psh2[i])*(psh1[i]-psh2[i]);
	}

	distance=sqrt(distance);

	return distance;
}

