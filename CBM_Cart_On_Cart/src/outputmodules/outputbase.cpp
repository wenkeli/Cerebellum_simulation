/*
 * outputbase.cpp
 *
 *  Created on: Jun 26, 2011
 *      Author: consciousness
 */

#include "../../includes/outputmodules/outputbase.h"

BaseOutput::BaseOutput(unsigned int numNC, float ts, float tsus)
{
	numNCIn=numNC;
	timeStepSize=ts;
	tsUnitInS=tsus;
}

BaseOutput::BaseOutput(ifstream &infile)
{
	infile.read((char *)&numNCIn, sizeof(unsigned int));
	infile.read((char *)&timeStepSize, sizeof(float));
	infile.read((char *)&tsUnitInS, sizeof(float));
}

BaseOutput::~BaseOutput()
{

}

void BaseOutput::exportState(ofstream &outfile)
{
	outfile.write((char *)&numNCIn, sizeof(unsigned int));
	outfile.write((char *)&timeStepSize, sizeof(float));
	outfile.write((char *)&tsUnitInS, sizeof(float));
}
