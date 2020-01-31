/*
 * innetsparsegrgo.cpp
 *
 *  Created on: Aug 9, 2011
 *      Author: consciousness
 */

#include "../../includes/innetmodules/innetsparsegrgo.h"

InNetSparseGRGO::InNetSparseGRGO(const bool *actInMF):InNet(actInMF)
{
	gGRIncGO=gGRIncGO*4.5;
	pfIncSC=0.0012;
	reinitializeVars();
	reinitializeCuda();

}

InNetSparseGRGO::InNetSparseGRGO(ifstream &infile, const bool *actInMF):InNet(infile, actInMF)
{
	gGRIncGO=gGRIncGO*4.5;
	pfIncSC=0.0012;
	reinitializeCuda();
}

void InNetSparseGRGO::reinitializeVars()
{
	stringstream output;

	cout<<"changing parameters"<<endl;
//	gMFIncGO=0;//gMFIncGO/2;

	cout<<"reconnecting using sparse GR to GO"<<endl;

	memset(numGOOutPerGR, 0, numGR*sizeof(int));
	for(int i=0; i<maxNumGOOutPerGR; i++)
	{
		for(int j=0; j<numGR; j++)
		{
			grConGROutGO[i][j]=numGO;
		}
	}

	assignGRGO(output, 200);//maxNumGRInPerGO*3/8); // /8;
	cout<<output.str()<<endl;
	cout<<"done"<<endl;
}

void InNetSparseGRGO::reinitializeCuda()
{
	cudaMemcpy2D(grConGROutGOGPU, grConGROutGOGPUP,
			grConGROutGO, numGR*sizeof(unsigned int),
			numGR*sizeof(unsigned int), maxNumGOOutPerGR, cudaMemcpyHostToDevice);
	cudaMemcpy(numGOOutPerGRGPU, numGOOutPerGR, numGR*sizeof(int), cudaMemcpyHostToDevice);
}
