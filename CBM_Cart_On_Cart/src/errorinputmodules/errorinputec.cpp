/*
 * errorinputec.cpp
 *
 *  Created on: May 25, 2011
 *      Author: consciousness
 */

#include "../../includes/errorinputmodules/errorinputec.h"

ECErrorInput::ECErrorInput(float maxE, float minE,
		float ts, float tsus, unsigned int errOnsetTSN):BaseErrorInput(maxE, minE, ts, tsus)
{
	errOnsetT=errOnsetTSN*ts*tsus;
	tsWindowInS=0.3*ts*tsus;
	errOnsetST=errOnsetT-tsWindowInS;
	errOnsetET=errOnsetT+tsWindowInS;
}

ECErrorInput::ECErrorInput(ifstream &infile):BaseErrorInput(infile)
{
	infile.read((char *)&errOnsetT, sizeof(float));
	infile.read((char *)&tsWindowInS, sizeof(float));
	infile.read((char *)&errOnsetST, sizeof(float));
	infile.read((char *)&errOnsetET, sizeof(float));

	cout<<"error module read: "<<maxErrSig<<" "<<minErrSig<<" "<<timeStepSize<<" "<<tsUnitInS<<" "
			<<errOnsetT<<" "<<tsWindowInS<<" "<<errOnsetST<<" "<<errOnsetET<<endl;
}

void ECErrorInput::calcActivity(unsigned int tsN, int trial)
{
	float timeInS=tsN*timeStepSize*tsUnitInS;
	currErrSig=0;

	if(timeInS>errOnsetST && timeInS<errOnsetET)
	{
		currErrSig=maxErrSig;
	}
}

void ECErrorInput::exportState(ofstream &outfile)
{
	BaseErrorInput::exportState(outfile);

	outfile.write((char *)&errOnsetT, sizeof(float));
	outfile.write((char *)&tsWindowInS, sizeof(float));
	outfile.write((char *)&errOnsetST, sizeof(float));
	outfile.write((char *)&errOnsetET, sizeof(float));
}
