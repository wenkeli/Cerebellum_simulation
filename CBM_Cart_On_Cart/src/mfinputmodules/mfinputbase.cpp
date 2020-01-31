/*
 * mfinputbase.cpp
 *
 *  Created on: Jun 23, 2011
 *      Author: consciousness
 */

#include "../../includes/mfinputmodules/mfinputbase.h"

BaseMFInput::BaseMFInput(unsigned int nmf, float ts, float tsus)
{
	numMF=nmf;
	timeStepSize=ts;
	tsUnitInS=tsus;
	apMF=new bool[numMF];
}

BaseMFInput::BaseMFInput(ifstream &infile)
{
	infile.read((char *)&numMF, sizeof(unsigned int));
	infile.read((char *)&timeStepSize, sizeof(float));
	infile.read((char *)&tsUnitInS, sizeof(float));

	apMF=new bool[numMF];
}

BaseMFInput::~BaseMFInput()
{
	delete[] apMF;
}

void BaseMFInput::exportState(ofstream &outfile)
{
	outfile.write((char *)&numMF, sizeof(unsigned int));
	outfile.write((char *)&timeStepSize, sizeof(float));
	outfile.write((char *)&tsUnitInS, sizeof(float));
}

void BaseMFInput::exportActDisp(vector<bool> &apRaster, int numCells)
{
	for(int i=0; i<numCells; i++)
	{
		apRaster[i]=apMF[i];
	}
}

