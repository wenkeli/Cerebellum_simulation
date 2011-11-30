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
	srand((unsigned int)time(NULL));

	for(int i=0; i<numCells; i++)
	{
		int cellInd1, cellInd2;

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

