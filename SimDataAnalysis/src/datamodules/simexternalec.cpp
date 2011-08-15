/*
 * simexternalec.cpp
 *
 *  Created on: Aug 15, 2011
 *      Author: consciousness
 */

#include "../../includes/datamodules/simexternalec.h"

SimExternalEC::SimExternalEC(ifstream &infile)
{
	infile.read((char *)&timeStepSize, sizeof(float));
	infile.read((char *)&tsUnitInS, sizeof(float));
}

