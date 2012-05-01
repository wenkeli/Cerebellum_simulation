/*
 * pshtravclustereucdist.cpp
 *
 *  Created on: Nov 29, 2011
 *      Author: consciousness
 */

#include "../../includes/analysismodules/pshtravclustereucdist.h"

EucDistPSHTravCluster::EucDistPSHTravCluster(PSHData *data, float thresh)
	:BasePSHTravCluster(data)
{
	threshP=thresh;
	generateDist();
}


bool EucDistPSHTravCluster::isDifferent(float *psh1, float *psh2)
{
	double distance;

	distance=calcEuclideanDist(psh1, psh2);

	if(distance>threshVal)
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

//	int *cellInd1List;
//	int *cellInd2List;

	CRandomSFMT0 randGen((int)time(NULL));
	unsigned int threshInd;

	//	randGen=new CRandomSFMT0((int)time(NULL));
	data=pshData->getData();

	distances.reserve(numCells);

//	cellInd1List=new int[numCells];
//	cellInd2List=new int[numCells];

	pshRow1=new float[pshNumBins];
	pshRow2=new float[pshNumBins];

	for(int i=0; i<numCells; i++)
	{
		int cellInd1, cellInd2;
		cellInd1=randGen.IRandom(0, numCells-1);
		cellInd2=randGen.IRandom(0, numCells-1);

		while(cellInd2==cellInd1)
		{
			cellInd1=randGen.IRandom(0, numCells-1);
			cellInd2=randGen.IRandom(0, numCells-1);
		}

		for(int j=0; j<pshNumBins; j++)
		{
			pshRow1[j]=data[j][cellInd1];
			pshRow2[j]=data[j][cellInd2];
		}
		distances[i]=calcEuclideanDist(pshRow1, pshRow2);
//		cout.precision(15);
//		cout<<cellInd1<<" "<<cellInd2<<" "<<fixed<<distances[i]<<endl;
	}

//	cout.precision(15);
//	for(int i=0; i<numCells; i++)
//	{
//		cout<<cellInd1List[i]<<" "<<cellInd2List[i]<<" "<<fixed<<distances[i]<<endl;
//	}

	cerr<<distances[0]<<endl;

	sort(&distances[0], &distances[numCells], less<double>());

	threshInd=floor(threshP*(numCells-1));
	threshVal=distances[threshInd];

	cerr<<distances[0]<<endl;
	cerr<<*(distances.begin())<<endl;

	cerr<<"ThreshVal, ind: "<<threshVal<<" "<<threshInd<<endl;
}

double EucDistPSHTravCluster::calcEuclideanDist(float *psh1, float *psh2)
{
	double distance;

	distance=0;

	for(int i=0; i<pshNumBins; i++)
	{
		distance+=(((double)psh1[i])-((double)psh2[i]))*(((double)psh1[i])-((double)psh2[i]));
	}

	distance=sqrt(distance);

	return distance;
}

