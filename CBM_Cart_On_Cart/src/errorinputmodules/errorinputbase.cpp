/*
 * errorinputbase.cpp
 *
 *  Created on: Jun 26, 2011
 *      Author: consciousness
 */

#include "../../includes/errorinputmodules/errorinputbase.h"

BaseErrorInput::BaseErrorInput(float maxE, float minE, float ts, float tsus)
{
	maxErrSig=maxE;
	minErrSig=minE;
	timeStepSize=ts;
	tsUnitInS=tsus;
}

BaseErrorInput::BaseErrorInput(ifstream &infile)
{
	infile.read((char *)&maxErrSig, sizeof(float));
	infile.read((char *)&minErrSig, sizeof(float));

	infile.read((char *)&timeStepSize, sizeof(float));
	infile.read((char *)&tsUnitInS, sizeof(float));
}

BaseErrorInput::~BaseErrorInput()
{

}

void BaseErrorInput::exportState(ofstream &outfile)
{
	outfile.write((char *)&maxErrSig, sizeof(float));
	outfile.write((char *)&minErrSig, sizeof(float));
	outfile.write((char *)&timeStepSize, sizeof(float));
	outfile.write((char *)&tsUnitInS, sizeof(float));
}
