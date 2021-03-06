/*
 * readinputs.cpp
 *
 *  Created on: Feb 15, 2011
 *      Author: consciousness
 */

#include "../includes/readinputs.h"

using namespace std;

bool readInputs(char *inputFileName)
{
	ifstream infile;

	infile.open(inputFileName, ios::binary);

	if(!infile.good() || !infile.is_open())
	{
		cerr<<"error opening file "<<inputFileName<<endl;

		return false;
	}

	infile.read((char *)ratesMFInputA, NUMMF*sizeof(float));
	infile.read((char *)ratesMFInputB, NUMMF*sizeof(float));

	return true;
}
