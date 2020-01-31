/*
 * externalbase.cpp
 *
 *  Created on: Jun 26, 2011
 *      Author: consciousness
 */
#include "../../includes/externalmodules/externalbase.h"

BaseExternal::BaseExternal(float ts, float tsus, BaseErrorInput **ems, BaseOutput **oms, unsigned int numMods)
{
	timeStepSize=ts;
	tsUnitInS=tsus;
	errorModules=ems;
	outputModules=oms;
	numModules=numMods;
}

BaseExternal::BaseExternal(ifstream &infile, BaseErrorInput **ems, BaseOutput **oms, unsigned int numMods)
{
	infile.read((char *)&timeStepSize, sizeof(float));
	infile.read((char *)&tsUnitInS, sizeof(float));

	errorModules=ems;
	outputModules=oms;
	numModules=numMods;
}

BaseExternal::~BaseExternal()
{

}

void BaseExternal::exportState(ofstream &outfile)
{
	outfile.write((char *)&timeStepSize, sizeof(float));
	outfile.write((char *)&tsUnitInS, sizeof(float));
}
