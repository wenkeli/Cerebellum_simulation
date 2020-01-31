/*
 * externaldummy.cpp
 *
 *  Created on: Jun 9, 2011
 *      Author: consciousness
 */
#include "../../includes/externalmodules/externaldummy.h"

DummyExternal::DummyExternal(float ts, float tsus, BaseErrorInput **ems, BaseOutput **oms, unsigned int numMods)
	:BaseExternal(ts, tsus, ems, oms, numMods)
{

}

DummyExternal::DummyExternal(ifstream &infile, BaseErrorInput **ems, BaseOutput **oms, unsigned int numMods):
		BaseExternal(infile, ems, oms, numMods)
{
	cout<<"external mod read: "<<timeStepSize<<" "<<tsUnitInS<<" "<<numMods<<endl;
}

void DummyExternal::run()
{

}

void DummyExternal::exportState(ofstream &outfile)
{
	BaseExternal::exportState(outfile);
}
