/*
 * errorinputcp.cpp
 *
 *  Created on: May 27, 2011
 *      Author: mhauskn
 */
#include "../../includes/errorinputmodules/errorinputcp.h"

CPErrorInput::CPErrorInput(float maxE, float minE, float ts, float tsus): BaseErrorInput(maxE, minE, ts, tsus)
{
}

CPErrorInput::CPErrorInput(ifstream &infile):BaseErrorInput(infile)
{
}

void CPErrorInput::exportState(ofstream &outfile)
{
	BaseErrorInput::exportState(outfile);
}

void CPErrorInput::calcActivity(unsigned int tsN, int trial)
{
	currErrSig = inError * maxErrSig;
}
