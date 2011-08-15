/*
 * simmfinputec.cpp
 *
 *  Created on: Aug 15, 2011
 *      Author: consciousness
 */

#include "../../includes/datamodules/simmfinputec.h"

SimMFInputEC::SimMFInputEC(ifstream &infile)
{
	infile.read((char *)&numMF, sizeof(unsigned int));
	infile.read((char *)&timeStepSize, sizeof(float));
	infile.read((char *)&tsUnitInS, sizeof(float));

	infile.read((char *)&threshDecay, sizeof(float));
	infile.read((char *)&contextMFProportion, sizeof(float));
	infile.read((char *)&csTonicMFProportion, sizeof(float));
	infile.read((char *)&csPhasicMFProportion, sizeof(float));
	infile.read((char *)&csStart, sizeof(float));
	infile.read((char *)&csEnd, sizeof(float));
	infile.read((char *)&csPhasicMFEnd, sizeof(float));

	mfType=new char[numMF];
	bgFreq=new float[numMF];
	incFreq=new float[numMF];
	incStart=new float[numMF];
	incEnd=new float[numMF];
	threshold=new float[numMF];

	infile.read((char *)mfType, numMF*sizeof(char));
	infile.read((char *)bgFreq, numMF*sizeof(float));
	infile.read((char *)incFreq, numMF*sizeof(float));
	infile.read((char *)incStart, numMF*sizeof(float));
	infile.read((char *)incEnd, numMF*sizeof(float));
	infile.read((char *)threshold, numMF*sizeof(float));

	cout<<"mf mod read: "<<numMF<<" "<<timeStepSize<<" "<<tsUnitInS<<" "<<threshDecay<<" "
			<<contextMFProportion<<" "<<csTonicMFProportion<<" "<<csPhasicMFProportion<<" "
			<<csStart<<" "<<csEnd<<" "<<csPhasicMFEnd<<" "<<(int)mfType[0]<<" "
			<<bgFreq[0]<<" "<<incFreq[0]<<" "<<incStart[0]<<" "<<incEnd[0]<<" "<<threshold[0]<<endl;
}

SimMFInputEC::~SimMFInputEC()
{
	delete[] mfType;
	delete[] bgFreq;
	delete[] incFreq;
	delete[] incStart;
	delete[] incEnd;
	delete[] threshold;
}

