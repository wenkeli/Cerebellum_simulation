/*
 * simerrorec.cpp
 *
 *  Created on: Aug 15, 2011
 *      Author: consciousness
 */

#include "../../includes/datamodules/simerrorec.h"

SimErrorEC::SimErrorEC(ifstream &infile)
{
	infile.read((char *)&maxErrSig, sizeof(float));
	infile.read((char *)&minErrSig, sizeof(float));

	infile.read((char *)&timeStepSize, sizeof(float));
	infile.read((char *)&tsUnitInS, sizeof(float));

	infile.read((char *)&errOnsetT, sizeof(float));
	infile.read((char *)&tsWindowInS, sizeof(float));
	infile.read((char *)&errOnsetST, sizeof(float));
	infile.read((char *)&errOnsetET, sizeof(float));

	cout<<"error module read: "<<maxErrSig<<" "<<minErrSig<<" "<<timeStepSize<<" "<<tsUnitInS<<" "
			<<errOnsetT<<" "<<tsWindowInS<<" "<<errOnsetST<<" "<<errOnsetET<<endl;
}
