/*
 * simoutputec.cpp
 *
 *  Created on: Aug 15, 2011
 *      Author: consciousness
 */

#include "../../includes/datamodules/simoutputec.h"

SimOutputEC::SimOutputEC(ifstream &infile)
{
	infile.read((char *)&numNCIn, sizeof(unsigned int));
	infile.read((char *)&timeStepSize, sizeof(float));
	infile.read((char *)&tsUnitInS, sizeof(float));

	infile.read((char *)&gForceDecay, sizeof(float));
	infile.read((char *)&gForce, sizeof(float));
	infile.read((char *)&gForceLeak, sizeof(float));
	infile.read((char *)&vForce, sizeof(float));
	infile.read((char *)&velocityDecay, sizeof(float));
	infile.read((char *)&velocity, sizeof(float));
	infile.read((char *)&positionDecay, sizeof(float));
	infile.read((char *)&position, sizeof(float));
}

